from PIL import Image,ImageDraw
import pywavefront



texturePath = "horse.tga"
modelPath = "horse.obj"

texture = Image.open(texturePath)
width, height = texture.size

scene = pywavefront.Wavefront(modelPath,collect_faces=True)

vertFormat = scene.mesh_list[0].materials[0].vertex_format.split('_')
#print(vertFormat)
if not scene.mesh_list[0].materials[0].has_uvs:
    raise ValueError("No UV indormation in model")

dim = scene.mesh_list[0].materials[0].vertex_size

uvList = [scene.mesh_list[0].materials[0].vertices[i*dim:i*dim+2] for i in range(len(scene.mesh_list[0].faces)*3)]


mask = Image.new('RGB', texture.size)

draw = ImageDraw.Draw(mask)

for uvTri in [[i,i+1,i+2] for i in range(0,len(scene.mesh_list[0].faces)*3,3)]:
    draw.polygon([ (uvList[vert][0]*width,uvList[vert][1]*height) for vert in uvTri], fill = 'white')

mask = mask.transpose(Image.FLIP_TOP_BOTTOM)
mask.save(texturePath.split('.')[0]+"_mask"+'.jpg')
print("done")