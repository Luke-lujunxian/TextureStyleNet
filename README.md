# TextureStyleNet
 
Early work for COMP5214 project

## Requirements
Install according to respective guide 

- [nvdiffrast](https://github.com/NVlabs/nvdiffrast)

put under root dir of this repo, weight in `models`

- [LinearStyleTransfer](https://github.com/sunshineatnoon/LinearStyleTransfer)

FYI, Test enviroment `enviroment.yaml`

## Run

Three main process
``` 
Transfer only
    textureStyleNet.py 
Using SPN    
    textureStyleNetSPN.py
Using SPN with mask indication
    textureStyleNetSPNMask.py
```

## Disclaimer

`util.py` is from [LinearStyleTransfer](https://github.com/sunshineatnoon/LinearStyleTransfer)

Pytorch 1.10+ port for `LinearStyleTransfer\libs\spn_t1` is from [AnyNet](https://github.com/mileyan/AnyNet)

## Reference

This project is inspired by the following works

SPN Net https://research.nvidia.com/publication/2017-12_learning-affinity-spatial-propagation-networks

LinearStyleTransfer https://github.com/sunshineatnoon/LinearStyleTransfer

3DStyleNet https://nv-tlabs.github.io/3DStyleNet/