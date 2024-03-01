import numpy as np
from scipy.interpolate import RBFInterpolator
import pyvista
from time import time
training_param=np.array([0.05939321535345923,0.07436704297351776,0.06424870384644796,0.05903948646972072,0.048128931940501433,0.06813047017599906,0.04938284901364233,0.09025957007038718,0.09672964844509264,0.04450973669432]).reshape(10,1)
testing_param=np.array([0.052674735268473,0.03954796157128028,0.06498617876092566,0.08194220994024401,0.0292693994518832,0.034820602480672674,0.0801738485856289,0.06843378514703735,0.025783671822798858,0.02463727616389697]).reshape(10,1)


u_train_true=np.zeros((10,100,5017))
u_test_true=np.zeros((10,100,5017))
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


u_train_true=u_train_true.reshape(10,-1)
u_test_true=u_test_true.reshape(10,-1)



rbf=RBFInterpolator(training_param,u_train_true)
start=time()
u_train_pred=rbf(training_param)
u_test_pred=rbf(testing_param)
end=time()

print("{:.2e}".format(end-start))
print("{:.2e}".format(np.linalg.norm(u_train_pred-u_train_true)/np.linalg.norm(u_train_true)))
print("{:.2e}".format(np.linalg.norm(u_test_pred-u_test_true)/np.linalg.norm(u_test_true)))