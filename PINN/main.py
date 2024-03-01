from deepxde.geometry.pointcloud import PointCloud
import deepxde as dde
import numpy as np
import pyvista
import meshio
import torch
from time import time

times=np.linspace(0,1,100,dtype=np.float32)
theta=np.array([0.05939321535345923,0.07436704297351776,0.06424870384644796,0.05903948646972072,0.048128931940501433,0.06813047017599906,0.04938284901364233,0.09025957007038718, 0.09672964844509264,0.04450973669432],dtype=np.float32)

reader = pyvista.get_reader('../Train/NS1/snapshots/truth_0_u.xdmf')
points=reader.read().points

points=np.array(points[:,0:2],dtype=np.float32)
num_points=len(points)

left_boundary=points[np.where(points[:,0]<1e-8)] #this is imposed explicitly
right_boundary=points[np.where(points[:,0]>2.499999)]   
bottom_boundary=points[np.where(points[:,1]<1e-8)]#this is imposed explicitly
circle_boundary=points[np.where((points[:,0]-0.4)**2+(points[:,1]-0.205)**2<0.05**2+0.00001)]
top_boundary=points[np.where(points[:,1]>0.40999)] #this is imposed explicitly
interior=points[np.where((points[:,0]>1e-8) & (points[:,0]<2.499999) & (points[:,1]>1e-8) & (points[:,1]<0.40999))]

num_interior=len(interior)
num_boundary=len(circle_boundary)


#left_normal=np.array([-1,0])
right_normal=np.array([1,0])
circle_normal=(circle_boundary-np.array([0.4,0.205]).reshape(1,-1))/np.linalg.norm((circle_boundary-np.array([0.4,0.205]).reshape(1,-1)),axis=1).reshape(-1,1)

#bottom_normal=np.array([0,-1])
#top_normal=np.array([0,1])

#boundary_points=np.vstack((left_boundary,right_boundary,bottom_boundary,top_boundary))
#boundary_normals=np.vstack((left_normal*np.ones((left_boundary.shape[0],1)),right_normal*np.ones((right_boundary.shape[0],1)),bottom_normal*np.ones((bottom_boundary.shape[0],1)),top_normal*np.ones((top_boundary.shape[0],1))))

points_full=np.concatenate((
    np.tile(theta.reshape(1,1,-1,1),(num_points,100,1,1)),
    np.tile(times.reshape(1,-1,1,1),(num_points,1,10,1)),
    np.tile(points.reshape(-1,1,1,2),(1,100,10,1))),axis=3).reshape(-1,4)



interior_full=np.concatenate((
    np.tile(theta.reshape(1,1,-1,1),(num_interior,100,1,1)),
    np.tile(times.reshape(1,-1,1,1),(num_interior,1,10,1)),
    np.tile(interior.reshape(-1,1,1,2),(1,100,10,1))),axis=3).reshape(-1,4)
#


boundary_full=np.concatenate((
    np.tile(theta.reshape(1,1,-1,1),(num_boundary,100,1,1)),
    np.tile(times.reshape(1,-1,1,1),(num_boundary,1,10,1)),
    np.tile(circle_boundary.reshape(-1,1,1,2),(1,100,10,1))),axis=3).reshape(-1,4)
#



circle_normal=np.tile(circle_normal.reshape(-1,1,1,2),(1,100,10,1)).reshape(-1,2)
pc = PointCloud(interior_full,boundary_full, circle_normal)



def Navier_Stokes_Equation(x, y):
    u= y[:,0:1]
    u = y[:,0:1]
    v = y[:,1:2]
    p = y[:,2:3]
    du_x = dde.grad.jacobian(y, x, i=0, j=0)
    du_y = dde.grad.jacobian(y, x, i=0, j=1)
    du_t = dde.grad.jacobian(y, x, i=0, j=2)
    dv_x = dde.grad.jacobian(y, x, i=1, j=0)
    dv_y = dde.grad.jacobian(y, x, i=1, j=1)
    dv_t = dde.grad.jacobian(y, x, i=1, j=2)
    dp_x = dde.grad.jacobian(y, x, i=2, j=0)
    dp_y = dde.grad.jacobian(y, x, i=2, j=1)
    du_xx = dde.grad.hessian(y, x, component=0, i=0, j=0)
    du_yy = dde.grad.hessian(y, x, component=0, i=1, j=1)
    dv_xx = dde.grad.hessian(y, x, component=1, i=0, j=0)
    dv_yy = dde.grad.hessian(y, x, component=1, i=1, j=1)
    continuity = du_x + dv_y
    x_momentum = du_t + (u * du_x + v * du_y) + dp_x - x[:,0] * (du_xx + du_yy)
    y_momentum = dv_t + (u * dv_x + v * dv_y) + dp_y - x[:,0] * (dv_xx + dv_yy)
    return [continuity, x_momentum, y_momentum]


data = dde.data.PDE(
    pc,
    Navier_Stokes_Equation,
    [dde.icbc.DirichletBC(pc, lambda x: 0, lambda _, on_boundary: on_boundary)],
    num_domain=num_interior,
    num_boundary=num_boundary)

net = dde.nn.FNN(
  [4] + [500] * 4 + [3], "sin", "Glorot uniform"
)

def transform(x, y):    
    u=y[:,0]
    v=y[:,1]
    u_new=x[:,1]*((0.41-x[:,3])*(x[:,3])*(x[:,2]*u)+1./0.042025*x[:,3]*(0.41 - x[:,3]))
    v_new=x[:,1]*(0.41-x[:,3])*(x[:,3])*(x[:,2]*v)
    return torch.concat((u_new.unsqueeze(-1),v_new.unsqueeze(-1),y[:,2].unsqueeze(-1)),axis=1)

net.apply_output_transform(transform)

model = dde.Model(data, net)



model.compile("adam", lr=0.001)


losshistory, train_state = model.train(iterations=100,display_every=1)

theta_train=np.array([0.05939321535345923,0.07436704297351776,0.06424870384644796,0.05903948646972072,0.048128931940501433,0.06813047017599906,0.04938284901364233,0.09025957007038718, 0.09672964844509264,0.04450973669432],dtype=np.float32)
theta_test=np.array([0.052674735268473,0.03954796157128028,0.06498617876092566,0.08194220994024401,0.0292693994518832,0.034820602480672674,0.0801738485856289,0.06843378514703735,0.025783671822798858,0.02463727616389697],dtype=np.float32)


u_train_true=np.zeros((10,100,5017))
u_test_true=np.zeros((10,100,5017))
u_train_pred=np.zeros((10,100,5017))
u_test_pred=np.zeros((10,100,5017))


for i in range(10):
    reader = pyvista.get_reader('../Train/NS1/snapshots/truth_{}_u.xdmf'.format(i))
    for j in range(100):
        reader.set_active_time_value(j)
        u_train_true[i,j,:]=np.linalg.norm(reader.read().point_data[reader.read().point_data.keys()[0]],axis=1)

for i in range(10):
    reader = pyvista.get_reader('../Test/NS1/snapshots/truth_{}_u.xdmf'.format(i))
    for j in range(100):
        reader.set_active_time_value(j)
        u_test_true[i,j,:]=np.linalg.norm(reader.read().point_data[reader.read().point_data.keys()[0]],axis=1)


start=time()
for i in range(10):
    u_train_pred[i,:]=net(torch.tensor(np.concatenate((
    np.tile(theta_train[i].reshape(1,1,-1,1),(num_points,100,1,1)),
    np.tile(times.reshape(1,-1,1,1),(num_points,1,1,1)),
    np.tile(points.reshape(-1,1,1,2),(1,100,1,1))),axis=3)).reshape(-1,4))[:,0].detach().numpy().reshape(5017,100).T


for i in range(10):
    u_test_pred[i,:]=net(torch.tensor(np.concatenate((
    np.tile(theta_test[i].reshape(1,1,-1,1),(num_points,100,1,1)),
    np.tile(times.reshape(1,-1,1,1),(num_points,1,1,1)),
    np.tile(points.reshape(-1,1,1,2),(1,100,1,1))),axis=3)).reshape(-1,4))[:,0].detach().numpy().reshape(5017,100).T

end=time()
print(end-start)


print(u_train_pred.shape)
print(u_train_true.shape)
print(u_test_pred.shape)
print(u_test_true.shape)

print(np.linalg.norm(u_train_pred-u_train_true)/np.linalg.norm(u_train_true))
print(np.linalg.norm(u_test_pred-u_test_true)/np.linalg.norm(u_test_true))
""

tmp=meshio.read("../data/cylinder.xdmf")
mesh_points=tmp.points
cells=tmp.cells

u=u_train_pred[0]
with meshio.xdmf.TimeSeriesWriter("test.xdmf") as writer:
    writer.write_points_cells(points, cells)
    for t in range(100):
        writer.write_data(t, point_data={"phi": u[t]})

