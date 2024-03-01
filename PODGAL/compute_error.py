import numpy as np

time=np.load('time.npy')
import pyvista

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

for i in range(10):
    reader = pyvista.get_reader('./NS1/online_solution_train_{}_u.xdmf'.format(i))
    for j in range(100):
        reader.set_active_time_value(j)
        u_train_pred[i,j,:]=np.linalg.norm(reader.read().point_data[reader.read().point_data.keys()[0]],axis=1)

for i in range(10):
    reader = pyvista.get_reader('./NS1/online_solution_test_{}_u.xdmf'.format(i))
    for j in range(100):
        reader.set_active_time_value(j)
        u_test_pred[i,j,:]=np.linalg.norm(reader.read().point_data[reader.read().point_data.keys()[0]],axis=1)

print("{:.2e}".format((time)))
print("{:.2e}".format(np.linalg.norm(u_train_true-u_train_pred)/np.linalg.norm(u_train_true)))
print("{:.2e}".format(np.linalg.norm(u_test_true-u_test_pred)/np.linalg.norm(u_test_true)))