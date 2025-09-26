
# CyberSims AI Art Workflow v0.3.3.2
# Author: Scabhx
> **Professional character portrait generation with pose control and face reference conditioning**

## Overview

This ComfyUI workflow implements a sophisticated 7-stage diffusion pipeline optimized for high-quality character portraits. The system combines multiple conditioning modalities (text, pose, face reference) with advanced upscaling techniques to produce consistent, detailed results.


### Pipeline Flow Overview
```
Input Images → Preprocessing  → Conditioning  → Generation  → Upscaling  → Output
     ↓              ↓             ↓            ↓           ↓         ↓
┌─────────┐   ┌─────────┐   ┌───────────┐  ┌─────────┐ ┌───────────┐ ┌─────────┐
│ Pose    │   │ OpenPose│   │ ControlNet│  │ KSampler│ │Real-ESRGAN│ │ Final   │
│ Face    │   │ CLIP    │   │ IPAdapter │  │ 512×768 │ │ 4× Scale  │ │1024×1536│
│ Prompts │   │ Vision  │   │ Text Enc  │  │ 30 steps│ │ Hi-Res    │ │ PNG     │
└─────────┘   └─────────┘   └───────────┘  └─────────┘ └───────────┘ └─────────┘
```

## Required Models

All models are **free to download** and available from public repositories:

### Core Models
- **Checkpoint**: `cyberrealistic_v90.safetensors`
  - Type: SD 1.5 photorealistic checkpoint
  - Source: CivitAI
  - Purpose: Primary generative backbone

### ControlNet
- **Model**: `control_v11p_sd15_openpose_fp16.safetensors`
  - Type: OpenPose ControlNet
  - Source: HuggingFace (lllyasviel)
  - Purpose: Pose guidance and structure control

### IPAdapter
- **IPAdapter**: `ip-adapter-plus-face_sd15.safetensors`
  - **CLIP Vision**: `CLIP-ViT-H-14-laion2B-s32B-b79K.safetensors`
  - Source: HuggingFace (h94)
  - Purpose: Face reference conditioning

### LoRA Models
- **PortraitMaster**: `PortraitMasterV1.safetensors`
  - Purpose: Portrait composition enhancement
  - Strengths: Model=0.61, CLIP=0.65

- **Hand Fix**: `hand 5.5.safetensors`
  - Purpose: Hand anatomy correction
  - Strengths: Model=0.56, CLIP=0.63

### Upscaler
- **Model**: `4x_NickelbackFS_72000_G.pth`
  - Type: Real-ESRGAN variant
  - Factor: 4× super-resolution
  - Purpose: Detail enhancement

## Technical Specifications

### Generation Parameters
| Parameter | Value | Purpose |
|-----------|-------|---------|
| Base Resolution | 512×768 | 2:3 portrait aspect ratio |
| Sampler | DPM++ 2M | High-quality second-order solver |
| Scheduler | Karras | Improved noise distribution |
| CFG Scale | 9.0 | Strong prompt adherence |
| Steps | 30 | Balanced quality/speed |
| Primary Denoise | 92% | Near-complete generation |
| Hi-res Denoise | 51% | Detail refinement only |

### System Requirements
- **VRAM**: 8-12GB recommended
- **Generation Time**: 2-3 minutes (RTX 3080-level)
- **ComfyUI**: Latest version
- **Python**: 3.10+ recommended

## Workflow Stages

### 1. Base Model Loading
```python
CheckpointLoaderSimple
├── Model: CyberRealistic v9.0
├── CLIP: Text encoder for prompts
└── VAE: Image encode/decode
```

### 2. LoRA Chain Application
```python
Sequential LoRA Loading:
Base Model → PortraitMaster → Hand Fix → Enhanced Model
```
- **Hierarchical Enhancement**: Each LoRA builds upon the previous
- **Modular Design**: Easy to add/remove LoRAs

### 3. Multi-Modal Conditioning

#### 3A. Text Conditioning (Hierarchical)
```python
Prompt Structure:
├── Tier 1: Main Character ("casual, happy,")
├── Tier 2: Scene Context ("photography, italy, sun, male model")
└── Tier 3: LoRA Enhancement (configurable)
```

#### 3B. Visual Conditioning (IPAdapter)
```python
IPAdapter Configuration:
├── Weight: 0.65 (moderate influence)
├── Scaling: K+mean(V) w/ C penalty
├── Preprocessing: 512×512 LANCZOS crop
└── Model: IP-Adapter Plus Face
```

#### 3C. Pose Conditioning (ControlNet)
```python
OpenPose Settings:
├── Body Detection: Enabled
├── Hand Detection: Enabled  
├── Face Detection: Enabled
├── Resolution: 1024px
└── Strength: 0.58 (0.1%-65% range)
```

### 4. Primary Generation
High-quality base image generation at 512×768 with full tri-modal conditioning.

### 5. Upscaling Pipeline
```python
Two-Stage Upscaling:
512×768 → [AI 4×] → 2048×3072 → [Bicubic] → 1024×1536
```
- **Stage 1**: AI super-resolution for detail enhancement
- **Stage 2**: Bicubic scaling to optimal hi-res resolution

### 6. Hi-Res Fix
Latent-space detail refinement using the same conditioning setup with reduced denoising.

### 7. Final Output
PNG export with metadata preservation and custom filename prefix.

## Key Features

### 🎯 **Tri-Modal Conditioning**
Simultaneous text, pose, and face guidance for precise control over generation.

### 🔄 **Staged Resolution Enhancement**
Multi-step upscaling prevents artifacts while maintaining coherence.

### 🧩 **Modular Prompt Architecture**
Hierarchical prompting allows easy modification without breaking the pipeline.

### ⚡ **Optimized Performance**
Balanced settings for quality vs. generation time.

### 🎨 **Production Ready**
Consistent results suitable for professional workflows.

## Usage Instructions

### 1. Setup
```bash
# Install required ComfyUI custom nodes
pip install comfyui-controlnet-aux
pip install comfyui-ipadapter-plus
pip install comfyui-essentials
```

### 2. Model Installation
Download all required models and place them in appropriate ComfyUI folders:
```
models/checkpoints/cyberrealistic_v90.safetensors
models/controlnet/control_v11p_sd15_openpose_fp16.safetensors
models/ipadapter/ip-adapter-plus-face_sd15.safetensors
models/clip_vision/CLIP-ViT-H-14-laion2B-s32B-b79K.safetensors
models/loras/PortraitMasterV1.safetensors
models/loras/hand 5.5.safetensors
models/upscale_models/4x_NickelbackFS_72000_G.pth
```

### 3. Workflow Loading
1. Load `CyberSims_0.3.3.2.json` in ComfyUI
2. Upload your reference images:
   - **Pose reference**: Any image with clear body pose
   - **Face reference**: Close-up portrait for facial features
3. Adjust prompts in the text nodes
4. Configure generation parameters if needed
5. Queue the workflow

### 4. Customization

#### Prompt Modification
```python
# Main Character (Node 61)
"casual, happy, confident"

# Scene Context (Node 62)  
"photography, italy, sunset, male model"

# Negative Prompts (Node 3)
"lowres, bad anatomy, bad hands, text, error..."
```

#### Parameter Tuning
- **IPAdapter Weight**: 0.4-0.8 (face influence)
- **ControlNet Strength**: 0.3-0.8 (pose adherence)  
- **CFG Scale**: 7-12 (prompt following)
- **Denoise Values**: Adjust for style preference

## Technical Innovations

### 1. **Advanced IPAdapter Integration**
- Face-specific model with optimized preprocessing
- Penalty-based scaling for better coherence
- Configurable influence throughout generation

### 2. **Intelligent Upscaling Strategy**
- AI upscaling followed by controlled downsampling
- Prevents common hi-res artifacts
- Maintains detail integrity across scales

### 3. **Hierarchical Conditioning**
- Modular prompt combination system
- Independent control over character vs. scene
- Extensible architecture for additional conditioning

### 4. **Optimized Sampling Pipeline**
- Karras scheduling for superior noise handling
- DPM++ 2M for stability and quality
- Calibrated denoising for each stage

## Troubleshooting

### Common Issues

**Out of Memory**
- Reduce batch size to 1
- Lower hi-res resolution
- Disable unnecessary LoRAs

**Poor Face Similarity**
- Increase IPAdapter weight
- Use higher quality face reference
- Adjust face crop positioning

**Pose Not Following**
- Increase ControlNet strength
- Use clearer pose reference
- Check OpenPose preprocessing output

**Generation Too Slow**
- Reduce step count to 20-25
- Use different sampler (Euler a)
- Lower hi-res resolution

## Performance Optimization

### Memory Usage
```python
Typical VRAM Usage:
├── Base Generation: ~6GB
├── ControlNet Processing: +2GB
├── IPAdapter: +1GB
├── Hi-res Pass: +3GB
└── Total Peak: ~12GB
```

### Speed Optimization
- **Batch Processing**: Generate multiple variations
- **Model Caching**: Keep models loaded between runs
- **Resolution Scaling**: Adjust based on needs

## Version History

### v0.3.3.2 (Current)
- Improved IPAdapter integration
- Optimized upscaling pipeline
- Enhanced LoRA compatibility
- Better error handling

## Contributing

This workflow is open for community improvements:

1. **Fork** the repository
2. **Modify** the workflow
3. **Test** thoroughly
4. **Submit** pull request with description

### Suggested Improvements
- Additional LoRA support
- Alternative upscaling methods
- Regional prompting integration
- Animation keyframe support

## License

This workflow is released under the **MIT License**. Free to use, modify, and distribute.


## Credits

- **Workflow Design**: scabnx(me)
- **Base Models**: Various open-source contributors
- **ComfyUI**: ComfyAnonymous and community
- **ControlNet**: lllyasviel
- **IPAdapter**: tencent-ailab, h94

## Support

For questions, issues, or feature requests:
- **GitHub Issues**: [Repository Issues Page]

---

**Happy Generating!**