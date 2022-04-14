import pywavefront

scene = pywavefront.Wavefront(
    "box.obj",
    create_materials=True,
    collect_faces=True,
)

print("Faces:", scene.mesh_list[0].faces)
print("Vertices:", scene.vertices)
print("Format:", scene.mesh_list[0].materials[0].vertex_format)
print("Vertices:", [scene.mesh_list[0].materials[0].vertices[i:i+8] for i in range(0,len(scene.mesh_list[0].materials[0].vertices),8)])