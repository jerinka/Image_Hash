from PIL import Image
import imagehash
hash = imagehash.average_hash(Image.open('V1_tamp.jpg'))
print('hash:',hash)