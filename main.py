# AFK Clicker the Shitty Game

# Objectives
# 1) Implement GUI for level up and shops.
# 4) "Animate" certain actions and reactions.

# Import necessary libraries
import pygame as pg
from OpenGL.GL import *
from OpenGL.GL.shaders import compileProgram,compileShader
import numpy as np
from objLoaderV4 import ObjLoader
import shaderLoaderV3
import pyrr

FONT_TEX_COORDS = {
    "!" :  [1.0/32,  15.0/16, 2.0/32,  14.0/16],
    "\"" : [2.0/32,  15.0/16, 3.0/32,  14.0/16],
    "#" :  [3.0/32,  15.0/16, 4.0/32,  14.0/16],
    "$" :  [4.0/32,  15.0/16, 5.0/32,  14.0/16],
    "%" :  [5.0/32,  15.0/16, 6.0/32,  14.0/16],
    "&" :  [6.0/32,  15.0/16, 7.0/32,  14.0/16],
    "\'" : [7.0/32,  15.0/16, 8.0/32,  14.0/16],
    "(" :  [8.0/32,  15.0/16, 9.0/32,  14.0/16],
    ")" :  [9.0/32,  15.0/16, 10.0/32, 14.0/16],
    "*" :  [10.0/32, 15.0/16, 11.0/32, 14.0/16],
    "+" :  [11.0/32, 15.0/16, 12.0/32, 14.0/16],
    "," :  [12.0/32, 15.0/16, 13.0/32, 14.0/16],
    "-" :  [13.0/32, 15.0/16, 14.0/32, 14.0/16],
    "." :  [14.0/32, 15.0/16, 15.0/32, 14.0/16],
    "/" :  [15.0/32, 15.0/16, 16.0/32, 14.0/16],
    "0" :  [16.0/32, 15.0/16, 17.0/32, 14.0/16],
    "1" :  [17.0/32, 15.0/16, 18.0/32, 14.0/16],
    "2" :  [18.0/32, 15.0/16, 19.0/32, 14.0/16],
    "3" :  [19.0/32, 15.0/16, 20.0/32, 14.0/16],
    "4" :  [20.0/32, 15.0/16, 21.0/32, 14.0/16],
    "5" :  [21.0/32, 15.0/16, 22.0/32, 14.0/16],
    "6" :  [22.0/32, 15.0/16, 23.0/32, 14.0/16],
    "7" :  [23.0/32, 15.0/16, 24.0/32, 14.0/16],
    "8" :  [24.0/32, 15.0/16, 25.0/32, 14.0/16],
    "9" :  [25.0/32, 15.0/16, 26.0/32, 14.0/16],
    ":" :  [26.0/32, 15.0/16, 27.0/32, 14.0/16],
    ";" :  [27.0/32, 15.0/16, 28.0/32, 14.0/16],
    "<" :  [28.0/32, 15.0/16, 29.0/32, 14.0/16],
    "=" :  [29.0/32, 15.0/16, 30.0/32, 14.0/16],
    ">" :  [30.0/32, 15.0/16, 31.0/32, 14.0/16],
    "?" :  [31.0/32, 15.0/16, 32.0/32, 14.0/16],

    "@" :  [0.0,     14.0/16, 1.0/32,  13.0/16],
    "A" :  [1.0/32,  14.0/16, 2.0/32,  13.0/16],
    "B" :  [2.0/32,  14.0/16, 3.0/32,  13.0/16],
    "C" :  [3.0/32,  14.0/16, 4.0/32,  13.0/16],
    "D" :  [4.0/32,  14.0/16, 5.0/32,  13.0/16],
    "E" :  [5.0/32,  14.0/16, 6.0/32,  13.0/16],
    "F" :  [6.0/32,  14.0/16, 7.0/32,  13.0/16],
    "G" :  [7.0/32,  14.0/16, 8.0/32,  13.0/16],
    "H" :  [8.0/32,  14.0/16, 9.0/32,  13.0/16],
    "I" :  [9.0/32,  14.0/16, 10.0/32, 13.0/16],
    "J" :  [10.0/32, 14.0/16, 11.0/32, 13.0/16],
    "K" :  [11.0/32, 14.0/16, 12.0/32, 13.0/16],
    "L" :  [12.0/32, 14.0/16, 13.0/32, 13.0/16],
    "M" :  [13.0/32, 14.0/16, 14.0/32, 13.0/16],
    "N" :  [14.0/32, 14.0/16, 15.0/32, 13.0/16],
    "O" :  [15.0/32, 14.0/16, 16.0/32, 13.0/16],
    "P" :  [16.0/32, 14.0/16, 17.0/32, 13.0/16],
    "Q" :  [17.0/32, 14.0/16, 18.0/32, 13.0/16],
    "R" :  [18.0/32, 14.0/16, 19.0/32, 13.0/16],
    "S" :  [19.0/32, 14.0/16, 20.0/32, 13.0/16],
    "T" :  [20.0/32, 14.0/16, 21.0/32, 13.0/16],
    "U" :  [21.0/32, 14.0/16, 22.0/32, 13.0/16],
    "V" :  [22.0/32, 14.0/16, 23.0/32, 13.0/16],
    "W" :  [23.0/32, 14.0/16, 24.0/32, 13.0/16],
    "X" :  [24.0/32, 14.0/16, 25.0/32, 13.0/16],
    "Y" :  [25.0/32, 14.0/16, 26.0/32, 13.0/16],
    "Z" :  [26.0/32, 14.0/16, 27.0/32, 13.0/16],
    "[" :  [27.0/32, 14.0/16, 28.0/32, 13.0/16],
    "\\" : [28.0/32, 14.0/16, 29.0/32, 13.0/16],
    "]" :  [29.0/32, 14.0/16, 30.0/32, 13.0/16],
    "^" :  [30.0/32, 14.0/16, 31.0/32, 13.0/16],
    "_" :  [31.0/32, 14.0/16, 32.0/32, 13.0/16],

    "`" :  [0.0,     13.0/16, 1.0/32,  12.0/16],
    "a" :  [1.0/32,  13.0/16, 2.0/32,  12.0/16],
    "b" :  [2.0/32,  13.0/16, 3.0/32,  12.0/16],
    "c" :  [3.0/32,  13.0/16, 4.0/32,  12.0/16],
    "d" :  [4.0/32,  13.0/16, 5.0/32,  12.0/16],
    "e" :  [5.0/32,  13.0/16, 6.0/32,  12.0/16],
    "f" :  [6.0/32,  13.0/16, 7.0/32,  12.0/16],
    "g'" : [7.0/32,  13.0/16, 8.0/32,  12.0/16],
    "h" :  [8.0/32,  13.0/16, 9.0/32,  12.0/16],
    "i" :  [9.0/32,  13.0/16, 10.0/32, 12.0/16],
    "j" :  [10.0/32, 13.0/16, 11.0/32, 12.0/16],
    "k" :  [11.0/32, 13.0/16, 12.0/32, 12.0/16],
    "l" :  [12.0/32, 13.0/16, 13.0/32, 12.0/16],
    "m" :  [13.0/32, 13.0/16, 14.0/32, 12.0/16],
    "n" :  [14.0/32, 13.0/16, 15.0/32, 12.0/16],
    "o" :  [15.0/32, 13.0/16, 16.0/32, 12.0/16],
    "p" :  [16.0/32, 13.0/16, 17.0/32, 12.0/16],
    "q" :  [17.0/32, 13.0/16, 18.0/32, 12.0/16],
    "r" :  [18.0/32, 13.0/16, 19.0/32, 12.0/16],
    "s" :  [19.0/32, 13.0/16, 20.0/32, 12.0/16],
    "t" :  [20.0/32, 13.0/16, 21.0/32, 12.0/16],
    "u" :  [21.0/32, 13.0/16, 22.0/32, 12.0/16],
    "v" :  [22.0/32, 13.0/16, 23.0/32, 12.0/16],
    "w" :  [23.0/32, 13.0/16, 24.0/32, 12.0/16],
    "x" :  [24.0/32, 13.0/16, 25.0/32, 12.0/16],
    "y" :  [25.0/32, 13.0/16, 26.0/32, 12.0/16],
    "z" :  [26.0/32, 13.0/16, 27.0/32, 12.0/16],
    "{" :  [27.0/32, 13.0/16, 28.0/32, 12.0/16],
    "|" :  [28.0/32, 13.0/16, 29.0/32, 12.0/16],
    "}" :  [29.0/32, 13.0/16, 30.0/32, 12.0/16]
}
class TextLine:
    def __init__(self, font, text, shader, fontsize, startPos, color):
        self.font = font
        self.text = text
        self.shader = shader
        self.vertices = []
        self.vertexCount = 0
        self.fontsize = fontsize
        self.startPos = startPos
        self.color = np.array(color, dtype=np.float32)

        glUseProgram(self.shader)
        self.vao = glGenVertexArrays(1)
        self.vbo = glGenBuffers(1)
        self.build_text()

    def build_text(self):
        self.vertices = []
        self.vertexCount = 0

        for i in range(len(self.text)):
            character = self.text[i]
            if character in FONT_TEX_COORDS:
                #top left pos
                self.vertices.append(self.startPos[0] + i * self.fontsize[0])
                self.vertices.append(self.startPos[1] + self.fontsize[1])
                #top left tex coord
                self.vertices.append(FONT_TEX_COORDS[character][0])
                self.vertices.append(FONT_TEX_COORDS[character][1] - 0.15/16)
                #top right pos
                self.vertices.append(self.startPos[0] + (i + 1) * self.fontsize[0])
                self.vertices.append(self.startPos[1] + self.fontsize[1])
                #top right tex coord
                self.vertices.append(FONT_TEX_COORDS[character][2])
                self.vertices.append(FONT_TEX_COORDS[character][1] - 0.15/16)
                #bottom right pos
                self.vertices.append(self.startPos[0] + (i + 1) * self.fontsize[0])
                self.vertices.append(self.startPos[1] - self.fontsize[1])
                #bottom right tex coord
                self.vertices.append(FONT_TEX_COORDS[character][2])
                self.vertices.append(FONT_TEX_COORDS[character][3] - 0.15/16)

                #bottom right pos
                self.vertices.append(self.startPos[0] + (i + 1) * self.fontsize[0])
                self.vertices.append(self.startPos[1] - self.fontsize[1])
                #bottom right tex coord
                self.vertices.append(FONT_TEX_COORDS[character][2])
                self.vertices.append(FONT_TEX_COORDS[character][3] - 0.15/16)
                #bottom left pos
                self.vertices.append(self.startPos[0] + i * self.fontsize[0])
                self.vertices.append(self.startPos[1] - self.fontsize[1])
                #bottom left tex coord
                self.vertices.append(FONT_TEX_COORDS[character][0])
                self.vertices.append(FONT_TEX_COORDS[character][3] - 0.15/16)
                #top left pos
                self.vertices.append(self.startPos[0] + i * self.fontsize[0])
                self.vertices.append(self.startPos[1] + self.fontsize[1])
                #top left tex coord
                self.vertices.append(FONT_TEX_COORDS[character][0])
                self.vertices.append(FONT_TEX_COORDS[character][1] - 0.15/16)
                self.vertexCount += 6

        self.vertices = np.array(self.vertices,dtype=np.float32)
        glUseProgram(self.shader)
        glBindVertexArray(self.vao)
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo)
        glBufferData(GL_ARRAY_BUFFER, self.vertices.nbytes, self.vertices, GL_STATIC_DRAW)

        glEnableVertexAttribArray(0)
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 16, ctypes.c_void_p(0))

        glEnableVertexAttribArray(1)
        glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 16, ctypes.c_void_p(8))

    def draw(self):
        glUseProgram(self.shader)
        self.font.use()
        glUniform3fv(glGetUniformLocation(self.shader, "color"), 1, self.color)
        glBindVertexArray(self.vao)
        glDrawArrays(GL_TRIANGLES, 0, self.vertexCount)

    def destroy(self):
        glDeleteBuffers(1, (self.vbo,))
        glDeleteVertexArrays(1, (self.vao,))

def createShader(vertexFilepath, fragmentFilepath):

    with open(vertexFilepath, 'r') as f:
        vertex_src = f.readlines()

    with open(fragmentFilepath, 'r') as f:
        fragment_src = f.readlines()

    shader = compileProgram(compileShader(vertex_src, GL_VERTEX_SHADER),
                            compileShader(fragment_src, GL_FRAGMENT_SHADER))

    return shader

class SimpleMaterial:
    def __init__(self, filepath):
        self.texture = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, self.texture)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        image = pg.image.load(f"{filepath}.png").convert_alpha()
        image_width,image_height = image.get_rect().size
        img_data = pg.image.tostring(image,'RGBA')
        glTexImage2D(GL_TEXTURE_2D,0,GL_RGBA8,image_width,image_height,0,GL_RGBA,GL_UNSIGNED_BYTE,img_data)
        glGenerateMipmap(GL_TEXTURE_2D)

    def use(self):
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D,self.texture)

    def destroy(self):
        glDeleteTextures(1, (self.texture,))


def load_image(filename, flip=False):
    img = pg.image.load(filename)
    img_data = pg.image.tobytes(img, "RGB", flip)
    w, h = img.get_size()
    return img_data, w, h

def load_cubemap_texture(filenames):
    # Generate a texture ID
    texture_id = glGenTextures(1)

    # Bind the texture as a cubemap
    glBindTexture(GL_TEXTURE_CUBE_MAP, texture_id)

    # Define texture parameters
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE)
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_NEAREST)
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MAG_FILTER, GL_LINEAR)

    # Define the faces of the cubemap
    faces = [GL_TEXTURE_CUBE_MAP_POSITIVE_X, GL_TEXTURE_CUBE_MAP_NEGATIVE_X,
             GL_TEXTURE_CUBE_MAP_POSITIVE_Y, GL_TEXTURE_CUBE_MAP_NEGATIVE_Y,
             GL_TEXTURE_CUBE_MAP_POSITIVE_Z, GL_TEXTURE_CUBE_MAP_NEGATIVE_Z]

    # Load and bind images to the corresponding faces
    for i in range(6):
        img_data, img_w, img_h = load_image(filenames[i], flip=False)
        glTexImage2D(faces[i], 0, GL_RGB, img_w, img_h, 0, GL_RGB, GL_UNSIGNED_BYTE, img_data)

    # Generate mipmaps
    glGenerateMipmap(GL_TEXTURE_CUBE_MAP)

    # Unbind the texture
    glBindTexture(GL_TEXTURE_CUBE_MAP, 0)

    return texture_id

def dealDamage(dmg, reward):
    global health
    global material_color_mob
    global coin

    health = health - dmg
    coin = coin + reward
    print("======DEBUG===================================")
    print("Damage: ", dmg)
    print("Health Bar: ", health)
    print("Coin: ", coin)
    material_color_mob = (1.3, 0.2, 0.2)

def damageHelper():
    global health
    global coin
    DPS = dmg_help*npc/FPS
    dealDamage(DPS, npc*(0.2+DPS))

def drawText(x, y, text):
    textSurface = font.render(text, True, (222, 222, 222), (0, 66, 0))
    textData = pg.image.tobytes(textSurface, "RGB", False)
    glWindowPos2d(x, y)
    glDrawPixels(textSurface.get_width(), textSurface.get_height(), GL_RGBA, GL_UNSIGNED_BYTE, textData)

# Initialize pygame
pg.init()

# Set up OpenGL context version
pg.display.gl_set_attribute(pg.GL_CONTEXT_MAJOR_VERSION, 3)
pg.display.gl_set_attribute(pg.GL_CONTEXT_MINOR_VERSION, 3)

# Create a window for graphics using OpenGL
pg.display.set_caption("Here’s another one of those low-quality, AFK, addictive “tapping” games – Play it while on the toilet (or in class)!! ")
width = 450
height = 700
screen = pg.display.set_mode((width, height), pg.OPENGL | pg.DOUBLEBUF)


glClearColor(0.3, 0.4, 0.5, 1.0)
glEnable(GL_DEPTH_TEST)

# Camera parameters
eye = (0, 0, 2)
target = (0, 0, 0)
up = (0, 1, 0)

fov = 45
aspect = width / height
near = 0.1
far = 10

view_mat = pyrr.matrix44.create_look_at(eye, target, up)
projection_mat = pyrr.matrix44.create_perspective_projection_matrix(fov, aspect, near, far)

# material properties
material_color = (1.355, 0.655, 0.204)
material_color_mob = (1.355, 0.655, 0.204)
light_pos = np.array([2, 2, 2, None], dtype=np.float32)
# last component is for light type (0: directional, 1: point) which is changed by radio button

# Write our shaders. We will write our vertex shader and fragment shader in a different file
shaderProgram = shaderLoaderV3.ShaderProgram("shaders/vert.glsl", "shaders/frag.glsl")
shaderProgram_skybox = shaderLoaderV3.ShaderProgram("shaders/skybox/vert_skybox.glsl", "shaders/skybox/frag_skybox.glsl")
shader2DText = createShader("shaders/text/vertex_2d_textured.txt",
                            "shaders/text/fragment_2d_textured.txt")

# ***** Lets load our objects*********************************************************************************
# ***** Enemy Character Setup ********************************************************************************
enemyList = ["objects/cat.obj",
             "objects/dragon.obj",
             "objects/wolf.obj",
             "objects/teapot.obj"]

index = 2
mob = ObjLoader(enemyList[index])
transMat_ENEMY = pyrr.matrix44.create_from_translation([0, -0.1, 0])
scalingMat_ENEMY = pyrr.matrix44.create_from_scale([1.2 / mob.dia, 1.2 / mob.dia, 1.2 / mob.dia])
modelMat_ENEMY = pyrr.matrix44.multiply(transMat_ENEMY, scalingMat_ENEMY)

# ***** Create VAO, VBO, and configure vertex attributes for object mob *****
# VAO
vao1 = glGenVertexArrays(1)
glBindVertexArray(vao1)

# VBO
vbo1 = glGenBuffers(1)
glBindBuffer(GL_ARRAY_BUFFER, vbo1)
glBufferData(GL_ARRAY_BUFFER, mob.vertices.nbytes, mob.vertices, GL_STATIC_DRAW)

# Configure vertex attributes for object 1
position_loc = 0
glGetAttribLocation(shaderProgram.shader, "position")
glVertexAttribPointer(position_loc, mob.size_position, GL_FLOAT, GL_FALSE, mob.stride,
                      ctypes.c_void_p(mob.offset_position))
glEnableVertexAttribArray(position_loc)

tex_coord_loc = 1
glBindAttribLocation(shaderProgram.shader, tex_coord_loc, "uv")
glVertexAttribPointer(tex_coord_loc, mob.size_texture, GL_FLOAT, GL_FALSE, mob.stride, ctypes.c_void_p(mob.offset_texture))
glEnableVertexAttribArray(tex_coord_loc)

normal_loc = 1
glGetAttribLocation(shaderProgram.shader, "normal")
glVertexAttribPointer(normal_loc, mob.size_normal, GL_FLOAT, GL_FALSE, mob.stride, ctypes.c_void_p(mob.offset_normal))
glEnableVertexAttribArray(normal_loc)


# ***** Player character Setup ******************************************************************************
PC = ObjLoader("objects/raymanModel.obj")
translation_mat_PC = pyrr.matrix44.create_from_translation(-2.5 * PC.center)
scaling_mat = pyrr.matrix44.create_from_scale([0.7 / PC.dia, 0.7 / PC.dia, -0.7 / PC.dia])
model_mat_PC = pyrr.matrix44.multiply(translation_mat_PC, scaling_mat)

# ***** Create VAO, VBO, and configure vertex attributes for object 1 *****
# VAO
vao = glGenVertexArrays(1)
glBindVertexArray(vao)

# VBO
vbo = glGenBuffers(1)
glBindBuffer(GL_ARRAY_BUFFER, vbo)
glBufferData(GL_ARRAY_BUFFER, PC.vertices.nbytes, PC.vertices, GL_STATIC_DRAW)

# Configure vertex attributes for object 1
position_loc = 0
glGetAttribLocation(shaderProgram.shader, "position")
glVertexAttribPointer(position_loc, PC.size_position, GL_FLOAT, GL_FALSE, PC.stride,
                      ctypes.c_void_p(PC.offset_position))
glEnableVertexAttribArray(position_loc)

tex_coord_loc = 1
glBindAttribLocation(shaderProgram.shader, tex_coord_loc, "uv")
glVertexAttribPointer(tex_coord_loc, PC.size_texture, GL_FLOAT, GL_FALSE, PC.stride, ctypes.c_void_p(PC.offset_texture))
glEnableVertexAttribArray(tex_coord_loc)

normal_loc = 1
glGetAttribLocation(shaderProgram.shader, "normal")
glVertexAttribPointer(normal_loc, PC.size_normal, GL_FLOAT, GL_FALSE, PC.stride, ctypes.c_void_p(PC.offset_normal))
glEnableVertexAttribArray(normal_loc)

# ***** Skybox detail setup *************************************************************************************

# Define the vertices of the quad.
quad_vertices = (
            # Position
            -1, -1,
             1, -1,
             1,  1,
             1,  1,
            -1,  1,
            -1, -1
)
vertices = np.array(quad_vertices, dtype=np.float32)

size_position = 2       # x, y, z
stride = size_position * 4
offset_position = 0
quad_n_vertices = len(vertices) // size_position  # number of vertices

# Create VA0 and VBO
vao_quad = glGenVertexArrays(1)
glBindVertexArray(vao_quad)            # Bind the VAO. That is, make it the active one.
vbo_quad = glGenBuffers(1)                  # Generate one buffer and store its ID.
glBindBuffer(GL_ARRAY_BUFFER, vbo_quad)     # Bind the buffer. That is, make it the active one.
glBufferData(GL_ARRAY_BUFFER, vertices.nbytes, vertices, GL_STATIC_DRAW)   # Upload the data to the GPU.

# Define the vertex attribute configurations
# we can either query the locations of the attributes in the shader like we did in our previous assignments
# or explicitly tell the shader that the attribute "position" corresponds to location 0.
# It is recommended to explicitly set the locations of the attributes in the shader than querying them.
# Position attribute
position_loc = 0
glBindAttribLocation(shaderProgram_skybox.shader, position_loc, "position")
glVertexAttribPointer(position_loc, size_position, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(offset_position))
glEnableVertexAttribArray(position_loc)

# **************************************************************************************************************
# ****** Load Textures for cubemap *****************************************************************************

cubemap_images = ["skybox/right.png", "skybox/left.png",
                  "skybox/top.png", "skybox/bottom.png",
                  "skybox/front.png", "skybox/back.png"]

cubemap_id = load_cubemap_texture(cubemap_images)

shaderProgram_skybox["cubeMapTex"] = 0
# **************************************************************************************************************
# ***** Game Variables *****************************************************************************************
pressing_L = pressing_M = pressing_R = False
coin = 1000
enemy_health = 55.5
health = 55.5
dmg = 2.0
dmg_help = 5
dmg_incr = 1
enemy_mult = 2.1863                 # increases enemy health by this ratio
npc = 0                             # number of NPCs helping the player, does damage automatically
FPS = 30

clock = pg.time.Clock()

font = pg.font.SysFont('arial', 55)


if (dmg < 20.0):
    dmg_incr = 1.5
if (dmg > 21.0 and dmg < 1000.0):
    dmg_incr = 2.5
if (dmg > 101.0 and dmg < 10000.0):
    dmg_incr = 3.33
# ***************************************************************************************************************

# Run a loop to keep the program running
draw = True
while draw:
    clock.tick(20)
    # damageHelper()
    for event in pg.event.get():
        if event.type == pg.QUIT:
            draw = False

    # pointer to mouse events
    buttons = pg.mouse.get_pressed(num_buttons=3)

    # when player left-clicks, attack and decrease enemy health
    if buttons[0]:
        if not pressing_L:
            pressing_L = True
            dealDamage(dmg, 2.02)
    else:
        pressing_L = False
        material_color_mob = (1.3, 0.65, 0.2)

    # when player middle-clicks, spawn in helper to automatically deal damage.
    if buttons[1]:
        if not pressing_M and coin > 499:
            pressing_M = True
            coin = coin - 500
            npc += 1
            print('NEW HELPER ARRIVED!')

        elif not pressing_M and coin < 500:
            pressing_R = True
            print('INSUFFICIENT! Coins needed: ', 500 - coin)
    else:
        pressing_M = False


    # when play right-clicks, upgrade damage and take away coins
    if buttons[2]:
        if not pressing_R and coin > 149:
            pressing_R = True
            dmg = dmg * dmg_incr
            coin = coin - 150
            material_color = (0.65, 1.35, 0.2)
            print('UPGRADED! Coins left: ', coin)

        elif not pressing_R and coin < 150:
            pressing_R = True
            material_color = (1.3, 0.2, 0.2)
            print('INSUFFICIENT! Coins needed: ', 150 - coin)
    else:
        pressing_R = False
        material_color = (1.35, 0.65, 0.2)

    # On enemy death
    if (health <= 0):
        health = enemy_health * enemy_mult  # Increase health of next monster
        enemy_health = health  # Update health for next time.
        coin += 100


    view_mat_without_translation = view_mat.copy()
    view_mat_without_translation[3][:3] = [0,0,0]
    inverseViewProjection_mat = pyrr.matrix44.inverse(pyrr.matrix44.multiply(view_mat_without_translation,projection_mat))

    glBindTexture(GL_TEXTURE_CUBE_MAP, cubemap_id)

    # drawText((width/2)-80, height-80, "%.4f" % round(health, 4))
    # drawText(90, 30, "coins: %.2f" % round(coin, 2))


    # Clear color buffer and depth buffer before drawing each frame
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    # ****************************************************************************************************

    # Set uniforms for material
    glUseProgram(shaderProgram.shader)
    shaderProgram["model_matrix"] = model_mat_PC
    shaderProgram["view_matrix"] = view_mat
    shaderProgram["projection_matrix"] = projection_mat
    shaderProgram["material_color"] = material_color
    shaderProgram["light_pos"] = light_pos

    glBindVertexArray(vao)
    glDrawArrays(GL_TRIANGLES, 0, PC.n_vertices)  # draw the object
    # ****************************************************************************************************

    # Set uniforms for material
    glUseProgram(shaderProgram.shader)
    shaderProgram["model_matrix"] = modelMat_ENEMY
    shaderProgram["view_matrix"] = view_mat
    shaderProgram["projection_matrix"] = projection_mat
    shaderProgram["material_color"] = material_color_mob
    shaderProgram["light_pos"] = light_pos

    glBindVertexArray(vao1)
    glDrawArrays(GL_TRIANGLES, 0, mob.n_vertices)  # draw the object

    # ******************* Draw the skybox ****************************************************************

    glDepthFunc(GL_LEQUAL)  # Change depth function so depth test passes when values are equal to depth buffer's content
    glUseProgram(shaderProgram_skybox.shader)  # being explicit even though the line below will call this function
    shaderProgram_skybox["invViewProjectionMatrix"] = inverseViewProjection_mat
    glBindVertexArray(vao_quad)
    glDrawArrays(GL_TRIANGLES,
                 0,
                 quad_n_vertices)  # Draw the triangle
    glDepthFunc(GL_LESS)  # Set depth function back to default
    # *************************************************************

    # Refresh the display to show what's been drawn
    pg.display.flip()


    glBindFramebuffer(GL_FRAMEBUFFER, 0)
    # glDisable(GL_DEPTH_TEST)
    # glDisable(GL_CULL_FACE)

    font = SimpleMaterial("skybox/font")

    textLines = []
    health_label = TextLine(font, "%.2f" % round(health, 2), shader2DText, [0.5, 0.5], [width/2, height/2], [0,0,0])
    textLines.append(health_label)

    for line in textLines:
        line.draw()

# Cleanup
glDeleteVertexArrays(1, [vao, vao1, vao_quad])
glDeleteBuffers(1, [vbo, vbo1, vbo_quad])
glDeleteProgram(shaderProgram.shader)

glDeleteProgram(shaderProgram_skybox.shader)

pg.quit()  # Close the graphics window
quit()  # Exit the program
