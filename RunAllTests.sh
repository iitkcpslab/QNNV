echo "* MILP Tests"

echo "** MNIST Small"
# generate types
python3 ReduceIntegerPart.py networks/mnist/mnist_small_2.nnet 4 4 --noreduce > /dev/null 
# Run tests
python3 RunPerfTest.py 0 100 networks/mnist/mnist_small_2.nnet types.json 0.075 60 --cbc --cbc_par --glpk

echo "** MNIST Deep"
# generate types
python3 ReduceIntegerPart.py networks/mnist/mnist10x10.nnet 4 4 --noreduce > /dev/null
# Run tests
python3 RunPerfTest.py 0 100 networks/mnist/mnist10x10.nnet types.json 0.075 3600 --cbc --cbc_par --glpk

echo "** MNIST Tall"
# generate types
python3 ReduceIntegerPart.py networks/mnist/mnist2x256.nnet 4 4 --noreduce > /dev/null
# Run tests
python3 RunPerfTest.py 0 100 networks/mnist/mnist2x256.nnet types.json 0.075 3600 --cbc --cbc_par --glpk

echo "** MNISTFC"
# generate types
python3 ReduceIntegerPart.py networks/mnist/vnn_2x256.nnet 3 5 --noreduce > /dev/null
# Run tests
python3 RunPerfTest.py 0 0 networks/mnist/vnn_2x256.nnet types.json 0 3600 --cbc --cbc_par --glpk --vnnfiles networks/mnist/prop_0_0.05.vnnlib,networks/mnist/prop_1_0.05.vnnlib,networks/mnist/prop_2_0.05.vnnlib,networks/mnist/prop_3_0.05.vnnlib,networks/mnist/prop_4_0.05.vnnlib,networks/mnist/prop_5_0.05.vnnlib,networks/mnist/prop_6_0.05.vnnlib,networks/mnist/prop_7_0.05.vnnlib,networks/mnist/prop_8_0.05.vnnlib,networks/mnist/prop_9_0.05.vnnlib,networks/mnist/prop_10_0.05.vnnlib,networks/mnist/prop_11_0.05.vnnlib,networks/mnist/prop_12_0.05.vnnlib,networks/mnist/prop_13_0.05.vnnlib,networks/mnist/prop_14_0.05.vnnlib

echo "** MNIST-C"
# generate types
python3 ReduceIntegerPart.py networks/mnist/mnist_exp.nnet 4 2 --noreduce > /dev/null
# Run tests
python3 RunPerfTest.py 0 100 networks/mnist/mnist_exp.nnet types.json 0.25 3600 --cbc --cbc_par --glpk
python3 RunPerfTest.py 100 100 networks/mnist/mnist_exp.nnet types.json 0.5 3600 --cbc --cbc_par --glpk
python3 RunPerfTest.py 200 100 networks/mnist/mnist_exp.nnet types.json 0.75 3600 --cbc --cbc_par --glpk
python3 RunPerfTest.py 300 100 networks/mnist/mnist_exp.nnet types.json 1 3600 --cbc --cbc_par --glpk

echo "** FASHION-C"
# generate types
python3 ReduceIntegerPart.py networks/mnist/fashion_exp.nnet 4 2 --noreduce > /dev/null
# Run tests
python3 RunPerfTest.py 0 100 networks/mnist/fashion_exp.nnet types.json 0.25 3600 --cbc --cbc_par --glpk --fashion
python3 RunPerfTest.py 100 100 networks/mnist/fashion_exp.nnet types.json 0.5 3600 --cbc --cbc_par --glpk --fashion
python3 RunPerfTest.py 200 100 networks/mnist/fashion_exp.nnet types.json 0.75 3600 --cbc --cbc_par --glpk --fashion
python3 RunPerfTest.py 300 100 networks/mnist/fashion_exp.nnet types.json 1 3600 --cbc --cbc_par --glpk --fashion

echo "** Collision Avoidance"
cd coav
python RunTests.py 4 4
cd ..

echo "** TwinStream"
cd twin
python RunTests.py 4 4
cd ..

echo "** ACAS Xu"
cd acasxu
python RunTests.py 4 4
cd ..

echo "* PySMT Tests"

echo "** MNIST Small"
python3 RunSmtPerfTest.py 0 100 nil networks/mnist/mnist_small_2.nnet 0.075 60

echo "** MNIST Deep"
python3 RunSmtPerfTest.py 0 100 nil networks/mnist/mnist10x10.nnet 0.075 3600

echo "** MNIST Tall"
python3 RunSmtPerfTest.py 0 100 nil networks/mnist/mnist2x256.nnet 0.075 3600

echo "** MNISTFC"
echo "| Property | Time | Verdict |"
echo "|-"
timeout 3600s python3 PysmtVerify.py networks/mnist/vnn_2x256.nnet --prop networks/mnist/prop_0_0.05.vnnlib
timeout 3600s python3 PysmtVerify.py networks/mnist/vnn_2x256.nnet --prop networks/mnist/prop_1_0.05.vnnlib
timeout 3600s python3 PysmtVerify.py networks/mnist/vnn_2x256.nnet --prop networks/mnist/prop_2_0.05.vnnlib
timeout 3600s python3 PysmtVerify.py networks/mnist/vnn_2x256.nnet --prop networks/mnist/prop_3_0.05.vnnlib
timeout 3600s python3 PysmtVerify.py networks/mnist/vnn_2x256.nnet --prop networks/mnist/prop_4_0.05.vnnlib
timeout 3600s python3 PysmtVerify.py networks/mnist/vnn_2x256.nnet --prop networks/mnist/prop_5_0.05.vnnlib
timeout 3600s python3 PysmtVerify.py networks/mnist/vnn_2x256.nnet --prop networks/mnist/prop_6_0.05.vnnlib
timeout 3600s python3 PysmtVerify.py networks/mnist/vnn_2x256.nnet --prop networks/mnist/prop_7_0.05.vnnlib
timeout 3600s python3 PysmtVerify.py networks/mnist/vnn_2x256.nnet --prop networks/mnist/prop_8_0.05.vnnlib
timeout 3600s python3 PysmtVerify.py networks/mnist/vnn_2x256.nnet --prop networks/mnist/prop_9_0.05.vnnlib
timeout 3600s python3 PysmtVerify.py networks/mnist/vnn_2x256.nnet --prop networks/mnist/prop_10_0.05.vnnlib
timeout 3600s python3 PysmtVerify.py networks/mnist/vnn_2x256.nnet --prop networks/mnist/prop_11_0.05.vnnlib
timeout 3600s python3 PysmtVerify.py networks/mnist/vnn_2x256.nnet --prop networks/mnist/prop_12_0.05.vnnlib
timeout 3600s python3 PysmtVerify.py networks/mnist/vnn_2x256.nnet --prop networks/mnist/prop_13_0.05.vnnlib
timeout 3600s python3 PysmtVerify.py networks/mnist/vnn_2x256.nnet --prop networks/mnist/prop_14_0.05.vnnlib

echo "** Collision Avoidance"
cd coav
python RunSmtTests.py
cd ..

echo "** TwinStream"
cd twin
python RunSmtTests.py
cd ..

echo "** ACAS Xu"
cd acasxu
python RunSmtTests.py
cd ..
