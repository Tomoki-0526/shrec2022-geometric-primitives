import os
import torch
from torch.utils.data import DataLoader
from utils import *
from model.models import *
from dataset import DatasetSHREC2022
from loss import *
from geomfitty import geom3d, fit3d
import cone_fit
from copy import copy

def ls_fit_shape(point_cloud, cls, params):
    ini_params = copy(params)
    
    if isinstance(params, torch.Tensor):
        #print(params)
        params = params.cpu().numpy()
    if isinstance(point_cloud, torch.Tensor):
        point_cloud = point_cloud.cpu().numpy()

    try:
        if cls == 0: # plane
            pass


        elif cls == 1: # cylinder
            radius = params[0]
            axis = params[1:4]
            point = params[4:7]

            initial_guess = geom3d.Cylinder(point, axis, radius)
            cylinder = fit3d.cylinder_fit(point_cloud, weights=None, initial_guess=initial_guess)
            if not isinstance(cylinder, geom3d.Cylinder):
                print("Cylinder: Least squares method failed")
            else:
                radius, axis, point = cylinder.radius, cylinder.direction, cylinder.anchor_point
                radius = np.array([radius])
                params = np.concatenate([radius, axis, point])

            # get the loss between the predicted and the regressed shape
            # if the loss is large, then the least square algorithm has failed to converge 
            # and we use the network output

        elif cls == 2: # sphere
            radius = params[0]
            center = params[1:4]

            initial_guess = geom3d.Sphere(center, radius)
            sphere = fit3d.sphere_fit(point_cloud, weights=None, initial_guess=initial_guess)
            if not isinstance(sphere, geom3d.Sphere):
                print("Sphere: Least squares method failed")
            else:
                radius, center = sphere.radius, sphere.center
                radius = np.array([radius])
                params = np.concatenate([radius, center])

        elif cls == 3: # cone
            theta = params[0]
            axis = params[1:4]
            vertex = params[4:7]
            initial_guess = cone_fit.Cone(theta, axis, vertex)
            cone = cone_fit.cone_fit(point_cloud, weights=None, initial_guess=initial_guess)
            if not isinstance(cone, cone_fit.Cone):
                print("Cone: Least squares method failed")
                #params = params
            else:
                theta, axis, vertex = cone.theta, cone.axis, cone.vertex
                theta = np.array([theta])
                params = np.concatenate([theta, axis, vertex])

        elif cls == 4: # torus
            R = params[0]
            r = params[1]
            axis = params[2:5]
            center = params[5:8]

            initial_guess = geom3d.Torus(center, axis, R, r)
            torus = fit3d.torus_fit(point_cloud, weights=None, initial_guess=initial_guess)
            if not isinstance(torus, geom3d.Torus):
                print("Torus: Least squares method failed")
            else:
                R, r, axis, center = torus.major_radius, torus.minor_radius, torus.direction, torus.center
                R = np.array([R])
                r = np.array([r])
                params = np.concatenate([R, r, axis, center])

    except RuntimeError:
        # scipy throws runtime error is the maximum number of iterations is exceeded
        params = ini_params
        
    
    return torch.tensor(params)

def regress_params(shape, net_in, batch, trans):
    if shape == 0:
        
        normal = pla_net(net_in)
        point = batch['means'].to(device) # device is global
        params = torch.cat([normal, point], dim=-1)
        params = torch.cat(PlaneLoss().transform_plane_outputs(params, trans), dim=-1)
        
    if shape == 1:
        
        params = cyl_net(net_in)
        
        r, axis, vertex = CylinderLoss().transform_cylinder_outputs(params, trans)
        r = r.unsqueeze(-1)
        params = torch.cat([r, axis, vertex], dim=-1)
    
    if shape == 2: 
        
        params = sph_net(net_in)
        r, center = SphereLoss().transform_sphere_outputs(params, trans)
        r = r.unsqueeze(-1)
        params = torch.cat([r, center], dim=-1)
        
    if shape == 3: 
        
        params = con_net(net_in)
        theta, axis, vertex = ConeLoss().transform_cone_outputs(params, trans)
        theta = theta.unsqueeze(-1)
        params = torch.cat([theta, axis, vertex], dim=-1)
        
    if shape == 4:
        
        params = tor_net(net_in)
        R, r, axis, center = TorusLoss().transform_torus_outputs(params, trans)    
        R = R.unsqueeze(-1)
        r = r.unsqueeze(-1)
        params = torch.cat([R, r, axis, center], dim=-1)
    
    return params

def distance_points_shape(shape_type, shape_params, initial_points):
    
    if isinstance(shape_params, torch.Tensor):
        shape_params = shape_params.cpu().numpy()
    
    if isinstance(initial_points, torch.Tensor):
        initial_points = initial_points.cpu().numpy()
    
    if shape_type == 0:
        
        normal = shape_params[:3]
        vertex = shape_params[3:]
        shape = geomfitty.geom3d.Plane(normal, vertex)
        
    elif shape_type == 1:
        
        radius = shape_params[0]
        axis = shape_params[1:4]
        vertex = shape_params[4:7]
        shape = geomfitty.geom3d.Cylinder(vertex, axis, radius)
        
    elif shape_type == 2:
        
        radius = shape_params[0]
        center = shape_params[1:4]
        shape = geomfitty.geom3d.Sphere(center, radius)
        
    elif shape_type == 3:
        
        theta = shape_params[0]
        axis = shape_params[1:4]
        vertex = shape_params[4:7]
        shape = cone_fit.Cone(theta, axis, vertex)
    
    elif shape_type == 4:

        Radius = shape_params[0]
        radius = shape_params[1]
        axis = shape_params[2:5]
        center = shape_params[5:8]
        shape = geomfitty.geom3d.Torus(center, axis, Radius, radius)
    else:
        print("NOT ACCESSED")
    
    distance = shape.distance_to_point(initial_points).mean(0)

    return distance

def calc_loss(shape_type, ls_shape_params, shape_params):
    ls_shape_params = ls_shape_params.unsqueeze(0).to(device).float()
    if shape_type == 0: #plane 
        loss = 0
    
    elif shape_type == 1: # cylinder
        loss = CylinderLoss()(ls_shape_params, shape_params, None)
        loss = sum(loss)
        
    elif shape_type == 2: # sphere
        loss = SphereLoss()(ls_shape_params, shape_params, None)
        loss = sum(loss)
        
    elif shape_type == 3: # cone
        loss = ConeLoss()(ls_shape_params, shape_params, None)
        loss = sum(loss)
        
    elif shape_type == 4: # torus
        loss = TorusLoss()(ls_shape_params, shape_params, None)
        loss = sum(loss)
        
    return loss

def save_shape_prediction(path, shape_type, shape_params):
    if path == "":
        return
    
    sizes = {
        "1": 6,
        "2": 7,
        "3": 4,
        "4": 7,
        "5": 8
    }

    assert sizes[str(shape_type+1)] == len(shape_params)
    with open(path, "w") as F:
        F.write(str(shape_type + 1) + "\n")
        for param in shape_params:
            F.write(str(param) + "\n")

def norm(axis):
    assert axis.shape[-1] == 3
    
    return (axis * axis).sum(-1).sqrt()

def normalize_axis(shape_type, params):
    # normalize the axis and normal vectors
    
    if shape_type == 0: #plane
        normal, point = params[:3] , params[3:6]
        normal = normal / norm(normal)
        return torch.cat([normal, point])
        
    elif shape_type == 1: #cylinder
        r, axis, vertex = params[0].unsqueeze(0), params[1:4], params[4:7]
        axis = axis / norm(axis)
        return torch.cat([r, axis, vertex])
        
    elif shape_type == 2: # sphere
        return params
        
    elif shape_type == 3: # cone
        theta, axis, vertex = params[0].unsqueeze(0), params[1:4], params[4:7]
        axis = axis / norm(axis)
        return torch.cat([theta, axis, vertex])
    
    elif shape_type == 4: # torus
        R, r, axis, center = params[0].unsqueeze(0), params[1].unsqueeze(0), params[2:5], params[5:8]
        axis = axis / norm(axis)
        return torch.cat([R, r, axis, center])

# configuring paths for data and checkpoints
data_path = "/home/szj/SHREC2022/dataset/test"
checkpoint_path = "/home/szj/shrec2022-geometric-primitives/checkpoints"
output_path = "/home/szj/SHREC2022/results/methods/M6/prediction_results"

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

pla_net = PlaneNet().to(device)
pla_net.load_state_dict(torch.load(plane_checkpoint))
pla_net.eval()

cyl_net = CylinderNet().to(device)
cyl_net.load_state_dict(torch.load(cylinder_checkpoint))
cyl_net.eval()

sph_net = SphereNet().to(device)
sph_net.load_state_dict(torch.load(sphere_checkpoint))
sph_net.eval()

con_net = ConeNet().to(device)
con_net.load_state_dict(torch.load(cone_checkpoint))
con_net.eval()

tor_net = TorusNet().to(device)
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
    print('processsing: pointCloud{}.txt'.format(i+1))

    _, cls_pts, batch = data
    cls_pts = cls_pts.transpose(2, 1)
    cls_pts = cls_pts.to(device).float()
    pred = cls_net(cls_pts)
    shape_type = pred.detach().max(1)[1].cpu().numpy()[0]

    minknet_input = create_input_batch(
        batch, 
        device=device,
        quantization_size=0.05
    )

    trans = batch['trans']
    initial_points = batch['initial_points'][0]

    shape_params = regress_params(shape_type, minknet_input, batch, trans)
    ls_shape_params = ls_fit_shape(initial_points, shape_type, shape_params.squeeze(0))
    ls_fit_loss = calc_loss(shape_type, ls_shape_params, shape_params)
    if ls_fit_loss < 50:
        shape_params = ls_shape_params
    else:
        shape_params = shape_params.squeeze(0)
    
    shape_params = normalize_axis(shape_type, shape_params)

    out_path = lambda i : f"{output_path}/pointCloud{i}_prediction.txt"
    save_shape_prediction(out_path(i+1), shape_type, shape_params.cpu().numpy())