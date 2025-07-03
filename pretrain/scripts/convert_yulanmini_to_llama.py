import math
import os
import shutil
import sys
from collections import defaultdict

import torch
import torch.utils.checkpoint
from tqdm import tqdm
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM, LlamaConfig, LlamaForCausalLM
from modeling_yulanmini import YuLanMiniModelForCausalLM
from configuration_yulanmini import YuLanMiniConfig


def get_target_state_dict(
    source_state_dict,
    alpha_mapping,
    rms_type="llama",
    method="rebalanced",
):

    count = defaultdict(int)
    target_state_dict = {}
    for key, value in tqdm(source_state_dict.items()):
        if "alpha" in key:
            if "gate_up" in key:
                count[key] += 2
            else:
                count[key] += 1

        elif ".weight" in key:
            alpha_key = None
            done = False
            orig_key_norm = value.norm()
            for k, v in alpha_mapping.items():
                if k + ".weight" in key:
                    alpha_key = key.replace(k + ".weight", v)
                    if alpha_key not in source_state_dict:
                        print(
                            f"Not found: {key} -> {alpha_key}. Just copying the weights without multiplying alpha."
                        )
                        target_state_dict[key] = value
                        done = True
                        break

                    alpha = source_state_dict[alpha_key]

                    if method == "rebalanced":
                        if rms_type == "gemma" and "norm" in key:
                            target_state_dict[
                                key] = value.float() * alpha - 1 + alpha
                        else:
                            target_state_dict[key] = (value * alpha).to(
                                torch.bfloat16)
                        done = True

                    count[alpha_key] -= 1
                    key_norm = target_state_dict[key].norm()
                    alpha_norm = source_state_dict[alpha_key].item()
                    print(">>>", key, orig_key_norm, alpha_norm, alpha_key,
                          key_norm)
                    break

            if not done:
                raise ValueError(f"Not found {key}")
        else:
            target_state_dict[key] = value

    for key, value in count.items():
        if value != 0:
            print("\033[91mNot found: " + key + " " + str(value) + "\033[0m")

    print(target_state_dict.keys())
    return target_state_dict


def rebalance_weights2(model_path):
    method = 'llama'
    target_model_path = model_path + "-" + method
    print(f"copying model to {target_model_path}")
    shutil.copytree(model_path, target_model_path,
                    dirs_exist_ok=True, ignore=shutil.ignore_patterns('global_step*'))  # copy includes optimizer
    print("copying done")

    source_model = YuLanMiniModelForCausalLM.from_pretrained(target_model_path,
                                                        trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(target_model_path)
    print(source_model)
    # results = source_model.generate(**tokenizer("User: Hello, how are you?\nAssistant:", return_tensors="pt"), max_new_tokens=10)
    # print(tokenizer.decode(results[0], skip_special_tokens=True))

    if os.path.exists(target_model_path + "/model.safetensors"):
        os.remove(target_model_path +
                  "/model.safetensors")  # prepare for save_pretrained

    if os.path.exists(target_model_path + "/model.safetensors.index.json"):
        os.remove(target_model_path + "/model.safetensors.index.json")
        os.remove(target_model_path + "/model-00001-of-00002.safetensors")
        os.remove(target_model_path + "/model-00002-of-00002.safetensors")

    target_config = LlamaConfig(
        attention_bias=True,
        attention_dropout=source_model.config.attention_dropout,
        bos_token_id=source_model.config.bos_token_id,
        eos_token_id=source_model.config.eos_token_id,
        head_dim=source_model.config.hidden_size //
        source_model.config.num_attention_heads,
        hidden_act=source_model.config.hidden_act,
        hidden_size=source_model.config.hidden_size,
        initializer_range=source_model.config.initializer_range,
        intermediate_size=source_model.config.intermediate_size,
        max_position_embeddings=source_model.config.max_position_embeddings,
        mlp_bias=False,
        num_attention_heads=source_model.config.num_attention_heads,
        num_hidden_layers=source_model.config.num_hidden_layers,
        num_key_value_heads=source_model.config.num_key_value_heads,
        pretraining_tp=1,
        rms_norm_eps=source_model.config.rms_norm_eps,
        rope_scaling=None,
        rope_theta=source_model.config.rope_theta,
        tie_word_embeddings=False,
        torch_dtype=torch.float32,
        use_cache=True,
        vocab_size=source_model.config.vocab_size,
    )

    alpha_mapping = {
        ".embed_tokens": ".embed_tokens_alpha",
        ".q_proj": ".q_proj_alpha",
        ".k_proj": ".k_proj_alpha",
        ".v_proj": ".v_proj_alpha",
        ".o_proj": ".o_proj_alpha",
        ".mlp.down_proj": ".down_proj_alpha",
        ".mlp.gate_proj": ".gate_up_proj_alpha",
        ".mlp.up_proj": ".gate_up_proj_alpha",
        ".input_layernorm": ".input_layernorm_alpha",
        ".post_attention_layernorm": ".post_attention_layernorm_alpha",
        ".norm": ".norm_alpha",
        "lm_head": "lm_head_alpha"
    }

    source_state_dict = {k: v.float() for k, v in source_model.state_dict().items()}
    print(source_state_dict.keys())
    state_dict = get_target_state_dict(source_state_dict,
                                       alpha_mapping)
    if not hasattr(source_model.config, "scale_depth"):
        source_model.config.scale_depth = 1.4
    state_dict["model.embed_tokens.weight"] = state_dict[
        "model.embed_tokens.weight"] * source_model.config.scale_emb
    for i in range(source_model.config.num_hidden_layers):
        state_dict[f"model.layers.{i}.self_attn.o_proj.bias"] = torch.zeros(
            (source_model.config.hidden_size, ),
            dtype=state_dict[f"model.layers.{i}.mlp.down_proj.weight"].dtype)
        state_dict[f"model.layers.{i}.self_attn.o_proj.weight"] = state_dict[
            f"model.layers.{i}.self_attn.o_proj.weight"] * source_model.config.scale_depth / math.sqrt(
                source_model.config.num_hidden_layers)
        state_dict[f"model.layers.{i}.mlp.down_proj.weight"] = state_dict[
            f"model.layers.{i}.mlp.down_proj.weight"] * source_model.config.scale_depth / math.sqrt(
                source_model.config.num_hidden_layers)

    target_model = LlamaForCausalLM(target_config)
    # target_model = source_model
    target_model.load_state_dict(state_dict)

    target_model = target_model.to(torch.bfloat16)
    target_model.save_pretrained(target_model_path)
    print(target_model_path)

    # test model
    results = target_model.generate(**tokenizer("User: Hello, how are you?\nAssistant:", return_tensors="pt"), max_new_tokens=10)
    print(tokenizer.decode(results[0], skip_special_tokens=True))


if __name__ == "__main__":
    rebalance_weights2(sys.argv[1].rstrip("/"))
