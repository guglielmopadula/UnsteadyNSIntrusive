Tests on the unsteady navier stokes dataset of a [RBnics Tutorial](https://github.com/RBniCS/RBniCS/blob/master/tutorials/19_navier_stokes_unsteady/tutorial_navier_stokes_unsteady_exact_1.py).

|Method                                     |Train error|Test Error|TIime  |
|-------------------------------------------|-----------|----------|-------|
|PodGalerkinSUPG(Nmax=15,netested_POD=13)   |8.9e-05    |2.31e-04  |5.8e+03|
|PodGalerkinSUPG(Nmax=15)                   |1.6e-06    |5.36e-05  |4.5e+03|
|PodGalerkinSUPG(Nmax=5)                    |9.1e-03    |3.26e-02  |3.9e+03|
|Tree                                       |0.0e+00    |1.22e-02  |2.1e-02|
|GPR                                        |1.8e-04    |5.05e-03  |1.19-01|
|RBF                                        |8.8e-17    |6.82e-03  |3.27-02|