from skimage import io
import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms

from PIL import Image
import glob

from Data_loader import RescaleT
from Data_loader import ToTensorLab
from Data_loader import SalObjDataset

from model import MYNet
from tqdm import tqdm

def normPRED(d):
	ma = torch.max(d)
	mi = torch.min(d)

	dn = (d-mi)/(ma-mi)

	return dn

def save_output(image_name,pred,d_dir):

	predict = pred
	predict = predict.squeeze()
	predict_np = predict.cpu().data.numpy()

	im = Image.fromarray(predict_np*255).convert('RGB')
	img_name = image_name.split("/")[-1]
	image = io.imread(image_name)
	imo = im.resize((image.shape[1],image.shape[0]),resample=Image.BILINEAR)

	aaa = img_name.split(".")
	bbb = aaa[0:-1]
	imidx = bbb[0]
	for i in range(1,len(bbb)):
		imidx = imidx + "." + bbb[i]

	imo.save(d_dir+imidx+'.png')

image_dir = "/Image_test/"
prediction_dir = "/images_save/"
model_dir = "/model_save/EMFINet.pth"

img_name_list = glob.glob(image_dir + '*.jpg')

test_salobj_dataset = SalObjDataset(img_name_list = img_name_list, lbl_name_list = [],edge_name_list = [], transform=transforms.Compose([RescaleT(256),ToTensorLab(flag=0)]))
test_salobj_dataloader = DataLoader(test_salobj_dataset, batch_size=1,shuffle=False,num_workers=0)

net = MYNet()
net.load_state_dict(torch.load(model_dir))
if torch.cuda.is_available():
	net.cuda()
net.eval()

for i_test, data_test in tqdm(enumerate(test_salobj_dataloader)):

	inputs_test = data_test['image']
	inputs_test = inputs_test.type(torch.FloatTensor)

	if torch.cuda.is_available():
		inputs_test = Variable(inputs_test.cuda())
	else:
		inputs_test = Variable(inputs_test)
  
	de,d1,d2,d3,d4,d5,d6,d7,d8,d9 = net(inputs_test)

	# normalization
	pred = d9[:,0,:,:]
	pred = normPRED(pred)

	# save results to test_results folder
	save_output(img_name_list[i_test],pred,prediction_dir)

	del de,d1,d2,d3,d4,d5,d6,d7,d8,d9