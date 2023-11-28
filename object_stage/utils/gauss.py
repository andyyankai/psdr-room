import numpy as np
import torch as th


def gauss1d(sigma):
	r = int(np.ceil(sigma * 3))
	k = 1 + 2*r
	x = np.arange(k, dtype=np.float32) - r
	g = np.exp(-x**2 / (2*sigma**2))
	return (g / g.sum()).astype(np.float32)

def torch_gauss2d(img, sigma):
	print(img.shape)
	im_size = (img.shape[1], img.shape[2])
	nc = img.shape[0]
	img = img.unsqueeze(0)
	g = gauss1d(sigma)
	g = th.tensor(g, device=img.device).unsqueeze(0)
	g = th.stack([g]*nc, 0)
	img = img.transpose(2,3)
	img = img.reshape((1, 3, -1))
	c = th.nn.functional.conv1d(img, g, stride=1, groups=nc, padding='same')
	c = c.reshape(1, 3, im_size[1], im_size[0])
	c = c.transpose(2,3)
	c = c.reshape((1, 3, -1))
	c = th.nn.functional.conv1d(c, g, stride=1, groups=nc, padding='same')
	c = c.reshape(1, 3, im_size[0], im_size[1])
	return c.squeeze()


if __name__ == '__main__':
	from PIL import Image
	from torchvision.transforms import ToTensor

	img = Image.open('test.png')
	xform = ToTensor()
	img = xform(img)[:3,:,:]

	for i in range(9):
		sigma = 2**(i)
		result = torch_gauss2d(img, sigma)

		result = result.permute(1,2,0).detach().cpu().numpy()
		result *= 255
		Image.fromarray(np.clip(result, 0, 255).astype(np.uint8)).save(f'blur_{sigma}.jpg')
