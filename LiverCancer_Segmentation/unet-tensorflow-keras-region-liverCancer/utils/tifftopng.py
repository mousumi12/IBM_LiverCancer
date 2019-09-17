from PIL import Image
import glob

for name in glob.glob('../datasets/train/img/0/*.tiff'):
    im = Image.open(name)
    name = str(name).rstrip(".tiff")
    im.save(name + '.png', 'PNG')

for name in glob.glob('../datasets/train/gt/0/*.tiff'):
    im = Image.open(name)
    name = str(name).rstrip(".tiff")
    im.save(name + '.png', 'PNG')

for name in glob.glob('../datasets/val/img/0/*.tiff'):
    im = Image.open(name)
    name = str(name).rstrip(".tiff")
    im.save(name + '.png', 'PNG')


for name in glob.glob('../datasets/val/gt/0/*.tiff'):
    im = Image.open(name)
    name = str(name).rstrip(".tiff")
    im.save(name + '.png', 'PNG')

for name in glob.glob('../datasets/test/val/img/0/*.tiff'):
    im = Image.open(name)
    name = str(name).rstrip(".tiff")
    im.save(name + '.png', 'PNG')

for name in glob.glob('../datasets/test/val/gt/0/*.tiff'):
    im = Image.open(name)
    name = str(name).rstrip(".tiff")
    im.save(name + '.png', 'PNG')


print("Conversion from tif/tiff to png completed!")
