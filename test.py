import torch
import numpy as np
import math
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer

def build_rotation(r):
    norm = torch.sqrt(r[:,0]*r[:,0] + r[:,1]*r[:,1] + r[:,2]*r[:,2] + r[:,3]*r[:,3])

    q = r / norm[:, None]

    R = torch.zeros((q.size(0), 3, 3), device='cuda')

    r = q[:, 0]
    x = q[:, 1]
    y = q[:, 2]
    z = q[:, 3]

    R[:, 0, 0] = 1 - 2 * (y*y + z*z)
    R[:, 0, 1] = 2 * (x*y - r*z)
    R[:, 0, 2] = 2 * (x*z + r*y)
    R[:, 1, 0] = 2 * (x*y + r*z)
    R[:, 1, 1] = 1 - 2 * (x*x + z*z)
    R[:, 1, 2] = 2 * (y*z - r*x)
    R[:, 2, 0] = 2 * (x*z - r*y)
    R[:, 2, 1] = 2 * (y*z + r*x)
    R[:, 2, 2] = 1 - 2 * (x*x + y*y)
    return R

def build_scaling_rotation(s, r):
    L = torch.zeros((s.shape[0], 3, 3), dtype=torch.float, device="cuda")
    R = build_rotation(r)

    L[:,0,0] = s[:,0]
    L[:,1,1] = s[:,1]
    L[:,2,2] = s[:,2]

    L = R @ L
    return L

def getWorld2View(R, t):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = R.transpose()
    Rt[:3, 3] = t
    Rt[3, 3] = 1.0
    return np.float32(Rt)

def getWorld2View2(R, t, translate=np.array([.0, .0, .0]), scale=1.0):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = R.transpose()
    Rt[:3, 3] = t
    Rt[3, 3] = 1.0

    C2W = np.linalg.inv(Rt)
    cam_center = C2W[:3, 3]
    cam_center = (cam_center + translate) * scale
    C2W[:3, 3] = cam_center
    Rt = np.linalg.inv(C2W)
    return np.float32(Rt)

def getProjectionMatrix(znear, zfar, fovX, fovY):
    tanHalfFovY = math.tan((fovY / 2))
    tanHalfFovX = math.tan((fovX / 2))

    top = tanHalfFovY * znear
    bottom = -top
    right = tanHalfFovX * znear
    left = -right

    P = torch.zeros(4, 4)

    z_sign = 1.0

    P[0, 0] = 2.0 * znear / (right - left)
    P[1, 1] = 2.0 * znear / (top - bottom)
    P[0, 2] = (right + left) / (right - left)
    P[1, 2] = (top + bottom) / (top - bottom)
    P[3, 2] = z_sign
    P[2, 2] = z_sign * zfar / (zfar - znear)
    P[2, 3] = -(zfar * znear) / (zfar - znear)
    return P

def inverse_sigmoid(x):
    return torch.log(x/(1-x))

def RGB2SH(rgb):
    C0 = 0.28209479177387814
    return (rgb - 0.5) / C0

#------- test the diff gauss rasterizer ------
def render():

    # ---- setup diff gauss rasterizer ----
    bg_color=torch.tensor([0,0,0],dtype=torch.float).cuda() 

    FoVx = np.pi/4
    FoVy = np.pi/4
    tanfovx = math.tan(FoVx*0.5)
    tanfovy = math.tan(FoVy*0.5)
    image_height = 128
    image_width = 128
    znear=0.01
    zfar=100
    
    R = np.eye(3)
    T = np.zeros((3,1)) 
        
    world_view_transform = torch.tensor(getWorld2View(R, T)).transpose(0, 1).cuda()
    projection_matrix = getProjectionMatrix(znear=znear, zfar=zfar, fovX=FoVx, fovY=FoVy).transpose(0,1).cuda()
    camera_center = world_view_transform.inverse()[3, :3]

    raster_settings = GaussianRasterizationSettings(
        image_height=image_height,
        image_width=image_width,
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=1.0,
        viewmatrix=world_view_transform,
        projmatrix=projection_matrix,
        sh_degree=0,
        campos=camera_center,
        prefiltered=False,
        debug=False
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    N=10
    pcd_points = np.random.randn((N,3))
    pcd_colors = np.random.rand((N,3))
    fused_point_cloud = torch.tensor(np.asarray(pcd_points)).float().cuda()
    fused_color = RGB2SH(torch.tensor(np.asarray(pcd_colors)).float().cuda())
    opacities = inverse_sigmoid(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))
    screenspace_points = torch.zeros_like(fused_point_cloud, dtype=torch.float, device="cuda")
    
    means3D = fused_point_cloud
    means2D = screenspace_points
    opacity = opacities

    rots = torch.zeros((fused_point_cloud.shape[0],4),device="cuda")
    rots[:,0] = 1
    scales = np.ones_like(fused_point_cloud, dtype=torch.float, device="cuda")
    rotations = rots
    cov3D_precomp = None

    shs = None
    colors_precomp = fused_color

    rendered_image, radii = rasterizer(
        means3D = means3D,
        means2D = means2D,
        shs = shs,
        colors_precomp = colors_precomp,
        opacities = opacity,
        scales = scales,
        rotations = rotations,
        cov3D_precomp = cov3D_precomp
    )

# test run of diff gaussian rasterizer (output depth as well?)
if __name__ == '__main__':
    render()
    