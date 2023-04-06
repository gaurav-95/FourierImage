import glob
import matplotlib.pyplot as plt
from PIL import Image
from skimage import io
from skimage.transform import resize
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import mean_squared_error as mse
from skimage.metrics import normalized_root_mse as nrmse
from skimage.metrics import peak_signal_noise_ratio as psnr

# Get the list of image files
list_images = glob.glob("fouriers/*")

# Load the two images
image1 = io.imread(list_images[0], as_gray=True)
image2 = io.imread(list_images[1], as_gray=True)

# Resize the second image to the size of the first image
image2_resized = resize(image2, image1.shape, anti_aliasing=True)

# Calculate the similarity scores
ssim_score = ssim(image1, image2_resized, data_range=1.0)
mse_score = mse(image1, image2_resized)
nrmse_score = nrmse(image1, image2_resized, normalization='min-max')
psnr_score = psnr(image1, image2_resized, data_range=1.0)

# Create a table of the similarity scores
table_data = [['Similarity Metric', 'Score'],
              ['SSIM', round(ssim_score, 4)],
              ['MSE', round(mse_score, 4)],
              ['NRMSE', round(nrmse_score, 4)],
              ['PSNR', round(psnr_score, 4)]]

fig, ax = plt.subplots()
ax.axis('off')
ax.axis('tight')
ax.table(cellText=table_data, colWidths=[0.5, 0.5], cellLoc='center', loc='center')
fig.tight_layout()

# Save the table as an image
fig.savefig('similarity_table.png')

# Show the table
plt.show()
