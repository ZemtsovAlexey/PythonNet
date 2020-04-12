# Import python imaging libs
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont
from PIL import ImageChops

# Import operating system lib
import os
import pathlib

# Import random generator
from random import randint

#------------------------------------ Cleanup ----------------------------------#
    				
#def Cleanup():    
#    # Delete ds_store file
#    if os.path.isfile(font_dir + '.DS_Store'):
#        os.unlink(font_dir + '.DS_Store')

#    for file in os.listdir(out_dir):
#        file_path = os.path.join(out_dir, file)
#        if os.path.isfile(file_path):
#            os.unlink(file_path)
#    return

def Cleanup():
    if os.path.isfile(font_dir + '.DS_Store'):
        os.unlink(font_dir + '.DS_Store')

    for root, dirs, files in os.walk(out_dir, topdown=False):
        for name in files:
            os.remove(os.path.join(root, name))
        for name in dirs:
            os.rmdir(os.path.join(root, name))
    
    return

#------------------------------ Generate Characters ----------------------------#

def CreateDirectories():
    for char in characters:
        pathlib.Path(out_dir + "/" + char).mkdir(parents=True, exist_ok=True)

    return

def GenerateCharacters():
	# Counter
	k = 1
	# Process the font files
	for dirname, dirnames, filenames in os.walk(font_dir):
		# For each font do
		for filename in filenames:
			# Get font full file path
			font_resource_file = os.path.join(dirname, filename)
		
			# For each character do
			for char in characters:
				# For each font size do
				for font_size in font_sizes:
					if font_size > 0:
						# For each background color do
						for background_color in background_colors:
							# Convert the character into unicode
							character = char#.encode('utf-8')
			
							# Create character image : 
							# Grayscale, image size, background color
							char_image = Image.new('L', (image_size, image_size), background_color)
			
							# Draw character image
							draw = ImageDraw.Draw(char_image)
			
							# Specify font : Resource file, font size
							font = ImageFont.truetype(font_resource_file, font_size)
			
							# Get character width and height
							(font_width, font_height) = font.getsize(character)

							w, h = draw.textsize(character, font=font)
							h += int(h*0.30)
							# Calculate x position
							x = (image_size - w)/2
			
							# Calculate y position
							y = (image_size - h)/2
			
							# Draw text : Position, String, 
							# Options = Fill color, Font
							draw.text((x, y), character, (245-background_color) + randint(0, 10) , font=font)
					
							# Final file name    				
							file_name = out_dir + character + "/" + \
							filename + '_fs_' + \
							str(font_size) + '_bc_' + \
							str(background_color) + '.' + \
							character + "_" + str(k) + '.png'
			
							# Save image
							char_image.save(file_name)

							a = Image.open(file_name);
							bg = Image.new(a.mode, a.size, a.getpixel((0,0)))
							diff = ImageChops.difference(a, bg)
							bbox = diff.getbbox()
							nimg = a.crop(bbox)
							nimg.save(file_name)
					
							# Print character file name
							print(file_name)
					
							# Increment counter
							k = k + 1
	return

#---------------------------------- Input and Output ---------------------------#

# Directory containing fonts
font_dir = './fonts'

# Output
out_dir = './images/fonts/'

#------------------------------------ Characters -------------------------------#

numbers = list('0123456789')
#small_letters = list('qwertyuiopasdfghjklzxcvbnm')
small_letters = list('ёйцукенгшщзхъфывапролджэячсмитьбю')
#capital_letters = list('QWERTYUIOPASDFGHJKLZXCVBNM')
capital_letters = list('ЁЙЦУКЕНГШЩЗХЪФЫВАПРОЛДЖЭЯЧСМИТЬБЮ')
    	
# Select characters
characters = numbers + small_letters + capital_letters

#------------------------------------- Colors ----------------------------------#

# Background color
white_colors = (215, 225, 235, 245)
#black_colors = (0, 10, 20, 30)
gray_colors = (135, 145, 155)
test_colours = [215]

background_colors = white_colors + gray_colors
    	
#-------------------------------------- Sizes ----------------------------------#

# Character sizes
small_sizes = (8, 12, 16)
medium_sizes = (20, 24, 28)
large_sizes = (32, 36, 40)

font_sizes = [30]# small_sizes + medium_sizes + large_sizes
    	
# Image size
image_size = 30

#-------------------------------------- Main -----------------------------------#

# Do cleanup
Cleanup()

CreateDirectories()

# Generate characters
GenerateCharacters()
