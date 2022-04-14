from ast import Interactive
from code import interact
from logging import error
from email.policy import strict
import imp
from logging import warning
import string
from turtle import shape
from xml.etree.ElementTree import PI
import pywavefront
import torch
import numpy as np
import nvdiffrast.torch as dr
import util
import math
import torchvision.transforms as transforms
from PIL import Image

def transform_pos(mtx, pos):
    t_mtx = torch.from_numpy(mtx).cuda() if isinstance(mtx, np.ndarray) else mtx
    posw = torch.cat([pos, torch.ones([pos.shape[0], 1]).cuda()], axis=1)
    return torch.matmul(posw, t_mtx.t())[None, ...]

def render(glctx, mtx, pos, pos_idx, uv, uv_idx, tex, resolution, enable_mip, max_mip_level):
    pos_clip = transform_pos(mtx, pos)
    #print(tex.shape)
    rast_out, rast_out_db = dr.rasterize(glctx, pos_clip, pos_idx, resolution=[resolution, resolution])

    if enable_mip:
        texc, texd = dr.interpolate(uv[None, ...], rast_out, uv_idx, rast_db=rast_out_db, diff_attrs='all')
        color = dr.texture(tex[None, ...], texc, texd, filter_mode='linear-mipmap-linear', max_mip_level=max_mip_level)
    else:
        texc, _ = dr.interpolate(uv[None, ...], rast_out, uv_idx)
        color = dr.texture(tex[None, ...], texc, filter_mode='linear')

    color = color * torch.clamp(rast_out[..., -1:], 0, 1) # Mask out background.
    return color

def modelLoader(path:str):
    ### Open model
    scene = pywavefront.Wavefront(path,collect_faces=True)

    vertFormat = scene.mesh_list[0].materials[0].vertex_format.split('_')
    #print(vertFormat)
    if not scene.mesh_list[0].materials[0].has_uvs:
        error(f"No UV information, vertex format is {vertFormat}")
        raise ValueError("No UV indormation in model")

    dim = scene.mesh_list[0].materials[0].vertex_size

    uvList = [scene.mesh_list[0].materials[0].vertices[i*dim:i*dim+2] for i in range(len(scene.mesh_list[0].faces)*3)]

    ### Prepare render
    pos_idx = torch.from_numpy(np.array(scene.mesh_list[0].faces).astype(np.int32)).cuda()
    vtx_pos = torch.from_numpy(np.array(scene.vertices).astype(np.float32)).cuda()
    uv_idx  = torch.from_numpy(np.array([[i,i+1,i+2] for i in range(0,len(scene.mesh_list[0].faces)*3,3)]).astype(np.int32)).cuda()
    vtx_uv  = torch.from_numpy(np.array(uvList).astype(np.float32)).cuda()

    return vtx_pos,pos_idx,  vtx_uv, uv_idx
    

def textureLoader(path:str,fineSize):
    texture = Image.open(path).convert("RGB")
    texture = texture.resize((fineSize,fineSize))
    texture = texture.transpose(Image.FLIP_LEFT_RIGHT)
    #if texture.mode == "RGBA":
    #    texture.load() # required for png.split()
    #    warning("Converting RGBA to RGB, alpha channel not supported")
    #    background = Image.new("RGB", texture.size, (0, 0, 0))
    #    background.paste(texture, mask=texture.split()[3]) # 3 is the alpha channel
    #    texture = np.asarray(background)
    #elif texture.mode == 'RGB':
    #    texture = np.asarray(texture)
    #else:
    #    error(f"{texture.mode} is not supported")
    #    raise ValueError("Color space need to be RGB or RGBA")
    #    
    #texture = np.flipud(texture) # to GL convention
    transform = transforms.Compose([
            transforms.Resize(fineSize),
            transforms.ToTensor()])
    return transform(texture).cuda()
#https://github.com/pywavefront/PyWavefront/issues/87

def randomViewRender(vtx_pos, pos_idx, vtx_uv, uv_idx, tex, views=6, display = False,size = 1024):
    ang = 0.0
    glctx = dr.RasterizeGLContext()
    results = []
    for _ in range(views):

        # Random rotation/translation matrix for optimization.
        r_rot = util.random_rotation_translation(0.25)

        # Smooth rotation for display.
        a_rot = np.matmul(util.rotate_x(-0.4), util.rotate_y(ang))
        dist = np.random.uniform(0.0, 48.5)

        proj  = util.projection(x=0.4, n=1.0, f=200.0)
        r_mv  = np.matmul(util.translate(0, -5, -3), r_rot)
        r_mvp = np.matmul(proj, r_mv).astype(np.float32)
        a_mv  = np.matmul(util.translate(0, -.5, -3), a_rot)
        a_mvp = np.matmul(proj, a_mv).astype(np.float32)

        ### render
        color = render(glctx, r_mvp, vtx_pos, pos_idx, vtx_uv, uv_idx, tex, size, True, 3)
        results.append(color)
        #with torch.no_grad():
        #    color = render(glctx, a_mvp, vtx_pos, pos_idx, vtx_uv, uv_idx, tex, 768, False, 3)[0].cpu().numpy()[::-1]
        #    util.display_image(color, size=1024, title='1')

        #ang += 0.01
    return tuple(results) 
