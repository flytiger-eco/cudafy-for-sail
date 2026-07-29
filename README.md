# cudafy-for-sail

A small toolkit for converting PPU original HGGC-based repositories to CUDA-compatible source trees.

## Supported repositories

- `actlize`: dispatches versioned ACTLIZE include conversion scripts.
- `deepgemm`: dispatches the DeepGEMM conversion script.
- `flash-attention`: dispatches versioned Flash-Attention conversion scripts.
- `flashmla`: dispatches the FlashMLA conversion script.


## Usage

Run from the repository root:

```bash
python3 cudafy.py --help
```

### ACTLIZE

```bash
python3 cudafy.py actlize --version=1.0.0 /path/to/actlize/include
```

Supported versions: `0.5.0`, `0.8.0`, `1.0.0`.

### DeepGEMM

```bash
python3 cudafy.py deepgemm /path/to/DeepGEMM
```

### Flash-Attention

```bash
python3 cudafy.py flash-attention --version=2.8.2 /path/to/flash-attention
```

Supported versions: `2.7.2`, `2.7.4`, `2.8.2`.

### FlashMLA

```bash
python3 cudafy.py flashmla /path/to/FlashMLA
```

