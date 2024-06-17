CopyMark
===================
PyTorch implementation of **CopyMark: A Standardized Benchmark for Copyright Detection of Diffusion Models** based on diffusers and ComfyUI.

CopyMark provides:

- A benchmark for copyright detection on Latent Diffusion Model, Stable Diffusion, and SDXL (SD3 coming soon!) (based on diffusers)

- A GUI for using the benchmark to detect copyright images in the training data of diffusion models

**Guidance:**

- **Reproducing Results:** Check [README.md](diffusers/) in `diffusers` directory.

- **Using CopyMark Utility:** Check [README.md](ui/) in `ui` directory.


The code is organized as follows:

```
CopyMark
│   README.md
│   requirements.txt    
│
└───diffusers           # benchmark on diffusers
│   └───assets          # data: images for case studies
│   └───copymark        # code: diffusers pipelines of copyright detection
│   └───datasets        # data: put the datasets here
│   └───experiments     # data: raw records of original experimental results in the paper
│   └───scripts         # code: scripts to run copyright detection methods
│   └───utils           # code: scripts to prepare datasets & generate metadata
│   
└───ui
│   └───custom_nodes
│       └───assets                      # data: metadata for the inference of copyright detection
│       └───diffusers_ui                # code: diffusers pipelines of copyright detection
│           │   copymark.py             # code: functions of copyright detection used by nodes_copymark.py
│           │   encode_diffusers.py     # code: functions of encoding images and text like diffusers
│           │   load_diffusers.py       # code: functions of loading modules like diffusers
│       │   nodes_copymark.py           # code: custom nodes for copyright detection
│       │   nodes_diffusers.py          # code: custom nodes for adapting diffusers to comfyui
│   ... (the same as comfyui)
```


