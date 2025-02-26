import os
import torch
from torch.utils.data import DataLoader
from utils import *
from model.models import *
from dataset import DatasetSHREC2022
from loss import *

# configuring paths for data and checkpoints
data_path = "/home/szj/SHREC2022/dataset/test"
checkpoint_path = "/home/szj/shrec2022-geometric-primitives/checkpoints"
output_path = "/home/szj/SHREC2022/results/methods/M9/prediction_results"

cls_checkpoint = os.path.join(checkpoint_path, "classification.pth")
plane_checkpoint = os.path.join(checkpoint_path,"plane.pth")
sphere_checkpoint = os.path.join(checkpoint_path,"sphere.pth")
cylinder_checkpoint = os.path.join(checkpoint_path,"cylinder.pth")
cone_checkpoint = os.path.join(checkpoint_path,"cone.pth")
torus_checkpoint = os.path.join(checkpoint_path,"torus.pth")

# device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using Device: {device}")

# disabling gradient tracking for better performance
torch.set_grad_enabled(False)

# loading models
cls_net = torch.nn.DataParallel(Classifier(num_classes=5)).to(device)
cls_net.load_state_dict(torch.load(cls_checkpoint))
cls_net.eval()

pla_net = PlaneRegressor().to(device)
pla_net.load_state_dict(torch.load(plane_checkpoint))
pla_net.eval()

cyl_net = CylinderRegressor().to(device)
cyl_net.load_state_dict(torch.load(cylinder_checkpoint))
cyl_net.eval()

sph_net = SphereRegressor().to(device)
sph_net.load_state_dict(torch.load(sphere_checkpoint))
sph_net.eval()

con_net = ConeRegressor().to(device)
con_net.load_state_dict(torch.load(cone_checkpoint))
con_net.eval()

tor_net = TorusRegressor().to(device)
tor_net.load_state_dict(torch.load(torus_checkpoint))
tor_net.eval()

print('Networks loaded')

# loading data
test_dataset = DatasetSHREC2022(
    root=data_path,
    npoints=2048,
    split='test',
    transform=valid_transforms
)
test_loader = DataLoader(
    test_dataset,
    batch_size=1,
    shuffle=False,
    collate_fn=minkowski_collate_eval,
    num_workers=8
)

print('Data loaded')

# inference
for i, data in enumerate(test_loader, 0):
    _, cls_pts, reg_data = data
    cls_pts = cls_pts.transpose(2, 1)
    cls_pts = cls_pts.to(device).float()
    pred = cls_net(cls_pts)
    pred_choice = pred.detach().max(1)[1].cpu().numpy()[0]

    minknet_input = create_input_batch(
        reg_data, 
        device=device,
        quantization_size=0.05
    )

    with open(os.path.join(output_path, f'pointCloud{i+1}_prediction.txt'), 'wt') as f:
        print(f'processing: pointCloud{i+1}_prediction.txt')
        f.write(str(pred_choice+1)+'\n')

        # plane
        if pred_choice == 0:
            pred_normal = pla_net(minknet_input)
            pred_point = reg_data['means'].to(device)
            params = torch.cat((pred_normal, pred_point), dim=-1)
            pred_normal, pred_point = PlaneLoss().transform_plane_outputs(params, reg_data['trans'])

            pred_normal = torch.squeeze(pred_normal).cpu().detach().numpy()
            pred_point = torch.squeeze(pred_point).cpu().detach().numpy()

            f.write(str(pred_normal[0])+'\n')
            f.write(str(pred_normal[1])+'\n')
            f.write(str(pred_normal[2])+'\n')
            f.write(str(pred_point[0])+'\n')
            f.write(str(pred_point[1])+'\n')
            f.write(str(pred_point[2])+'\n')
        
        # cylinder
        elif pred_choice == 1:
            params = cyl_net(minknet_input)
            pred_radius, pred_axis, pred_point = CylinderLoss().transform_cylinder_outputs(params, reg_data['trans'])
            
            pred_radius = torch.squeeze(pred_radius).cpu().detach().numpy()
            pred_axis = torch.squeeze(pred_axis).cpu().detach().numpy()
            pred_point = torch.squeeze(pred_point).cpu().detach().numpy()

            f.write(str(pred_radius)+'\n')
            f.write(str(pred_axis[0])+'\n')
            f.write(str(pred_axis[1])+'\n')
            f.write(str(pred_axis[2])+'\n')
            f.write(str(pred_point[0])+'\n')
            f.write(str(pred_point[1])+'\n')
            f.write(str(pred_point[2])+'\n')

        # sphere
        elif pred_choice == 2:
            params = sph_net(minknet_input)
            pred_radius, pred_center = SphereLoss().transform_sphere_outputs(params, reg_data['trans'])

            pred_radius = torch.squeeze(pred_radius).cpu().detach().numpy()
            pred_center = torch.squeeze(pred_center).cpu().detach().numpy()

            f.write(str(pred_radius)+'\n')
            f.write(str(pred_center[0])+'\n')
            f.write(str(pred_center[1])+'\n')
            f.write(str(pred_center[2])+'\n')

        # cone
        elif pred_choice == 3:
            params = con_net(minknet_input)
            pred_theta, pred_axis, pred_vertex = ConeLoss().transform_cone_outputs(params, reg_data['trans'])

            pred_theta = torch.squeeze(pred_theta).cpu().detach().numpy()
            pred_axis = torch.squeeze(pred_axis).cpu().detach().numpy()
            pred_vertex = torch.squeeze(pred_vertex).cpu().detach().numpy()

            f.write(str(pred_theta)+'\n')
            f.write(str(pred_axis[0])+'\n')
            f.write(str(pred_axis[1])+'\n')
            f.write(str(pred_axis[2])+'\n')
            f.write(str(pred_vertex[0])+'\n')
            f.write(str(pred_vertex[1])+'\n')
            f.write(str(pred_vertex[2])+'\n')

        # torus
        else:
            assert pred_choice == 4
            params = tor_net(minknet_input)
            pred_major, pred_minor, pred_axis, pred_center = TorusLoss().transform_torus_outputs(params, reg_data['trans'])

            pred_major = torch.squeeze(pred_major).cpu().detach().numpy()
            pred_minor = torch.squeeze(pred_minor).cpu().detach().numpy()
            pred_axis = torch.squeeze(pred_axis).cpu().detach().numpy()
            pred_center = torch.squeeze(pred_center).cpu().detach().numpy()

            f.write(str(pred_major)+'\n')
            f.write(str(pred_minor)+'\n')
            f.write(str(pred_axis[0])+'\n')
            f.write(str(pred_axis[1])+'\n')
            f.write(str(pred_axis[2])+'\n')
            f.write(str(pred_center[0])+'\n')
            f.write(str(pred_center[1])+'\n')
            f.write(str(pred_center[2])+'\n')