import sys
import re
from pathlib import Path


def read_file(path):
    with open(path, 'r') as f:
        return f.read()


def write_file(path, content):
    with open(path, 'w') as f:
        f.write(content)


def ensure_qk_norm_field(content):
    if 'use_qk_norm:' in content:
        return content
    
    pattern = r'(attention_config:\n(?:.*\n)*?)(    attention_implementation:)'
    qk_norm_line = '      use_qk_norm: false\n'
    return re.sub(pattern, r'\1' + qk_norm_line + r'\2', content)


def set_qk_norm(content, norm_type, head_dim):
    configs = {
        'layer_norm': f"""      use_qk_norm: true
      qk_norm_config:
        norm_type: layer_norm
        config:
          normalized_shape: {head_dim}
          eps: 1.0e-05""",
        'pytorch_rms_norm': f"""      use_qk_norm: true
      qk_norm_config:
        norm_type: pytorch_rms_norm
        config:
          normalized_shape: {head_dim}
          eps: 1.0e-05""",
        'rms_norm': f"""      use_qk_norm: true
      qk_norm_config:
        norm_type: rms_norm
        config:
          dim: {head_dim}
          eps: 1.0e-05"""
    }
    
    return re.sub(r'      use_qk_norm: false', configs[norm_type], content)


def convert_norms(content, norm_type, n_embd):
    if norm_type == 'pytorch_rms_norm':
        template = 'norm_type: pytorch_rms_norm\n      config:\n        normalized_shape: {}\n        eps: 1.0e-05'
    else:
        template = 'norm_type: rms_norm\n      config:\n        dim: {}\n        eps: 1.0e-05'
    
    patterns = ['attention_norm_config', 'ffn_norm_config', 'lm_head_norm_config']
    
    for pattern_name in patterns:
        pattern = rf'{pattern_name}:\s+norm_type: [^\n]+\n\s+config:\s+[^\n]+\n\s+eps: [^\n]+'
        replacement = f'{pattern_name}:\n      ' + template.format(n_embd)
        content = re.sub(pattern, replacement, content)
    
    return content


def extract_model_params(content):
    n_embd = int(re.search(r'n_embd: (\d+)', content).group(1))
    n_head_q = int(re.search(r'n_head_q: (\d+)', content).group(1))
    return n_embd, n_head_q, n_embd // n_head_q


def main():
    if len(sys.argv) != 2:
        print("Usage: python generate_configs.py <baseline_config.yaml>")
        sys.exit(1)
    
    baseline_path = Path(sys.argv[1])
    baseline_content = ensure_qk_norm_field(read_file(baseline_path))
    n_embd, n_head_q, head_dim = extract_model_params(baseline_content)
    
    print(f"Detected: n_embd={n_embd}, n_head_q={n_head_q}, head_dim={head_dim}")
    
    configs = [
        ('qknorm_layernorm', lambda c: set_qk_norm(c, 'layer_norm', head_dim)),
        ('qknorm_pytorch_rms', lambda c: convert_norms(set_qk_norm(c, 'pytorch_rms_norm', head_dim), 'pytorch_rms_norm', n_embd)),
        ('qknorm_rms', lambda c: convert_norms(set_qk_norm(c, 'rms_norm', head_dim), 'rms_norm', n_embd)),
        ('no_qknorm_pytorch_rms', lambda c: convert_norms(c, 'pytorch_rms_norm', n_embd)),
    ]
    
    for suffix, transform in configs:
        output_path = baseline_path.parent / f"{baseline_path.stem}_{suffix}.yaml"
        write_file(output_path, transform(baseline_content))
        print(f"Generated: {output_path}")


if __name__ == "__main__":
    main()