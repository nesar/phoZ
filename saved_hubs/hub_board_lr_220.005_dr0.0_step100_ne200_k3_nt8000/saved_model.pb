ћ┤
вм
:
Add
x"T
y"T
z"T"
Ttype:
2	
B
AssignVariableOp
resource
value"dtype"
dtypetypeѕ
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
8
Const
output"dtype"
valuetensor"
dtypetype
,
Exp
x"T
y"T"
Ttype:

2
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
=
Mul
x"T
y"T
z"T"
Ttype:
2	љ
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
~
RandomUniform

shape"T
output"dtype"
seedint "
seed2int "
dtypetype:
2"
Ttype:
2	ѕ
@
ReadVariableOp
resource
value"dtype"
dtypetypeѕ
E
Relu
features"T
activations"T"
Ttype:
2	
:
Sub
x"T
y"T
z"T"
Ttype:
2	
q
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshapeѕ
9
VarIsInitializedOp
resource
is_initialized
ѕ*1.13.12
b'unknown'8оњ
d
XPlaceholder*
dtype0*'
_output_shapes
:         *
shape:         
Ъ
-dense/kernel/Initializer/random_uniform/shapeConst*
dtype0*
_output_shapes
:*
_class
loc:@dense/kernel*
valueB"       
Ћ
+dense/kernel/Initializer/random_uniform/minConst*
_class
loc:@dense/kernel*
valueB 2}&ё╗┼┘┐*
dtype0*
_output_shapes
: 
Ћ
+dense/kernel/Initializer/random_uniform/maxConst*
dtype0*
_output_shapes
: *
_class
loc:@dense/kernel*
valueB 2}&ё╗┼┘?
╠
5dense/kernel/Initializer/random_uniform/RandomUniformRandomUniform-dense/kernel/Initializer/random_uniform/shape*
dtype0*
_output_shapes

: *
T0*
_class
loc:@dense/kernel
╬
+dense/kernel/Initializer/random_uniform/subSub+dense/kernel/Initializer/random_uniform/max+dense/kernel/Initializer/random_uniform/min*
T0*
_class
loc:@dense/kernel*
_output_shapes
: 
Я
+dense/kernel/Initializer/random_uniform/mulMul5dense/kernel/Initializer/random_uniform/RandomUniform+dense/kernel/Initializer/random_uniform/sub*
T0*
_class
loc:@dense/kernel*
_output_shapes

: 
м
'dense/kernel/Initializer/random_uniformAdd+dense/kernel/Initializer/random_uniform/mul+dense/kernel/Initializer/random_uniform/min*
T0*
_class
loc:@dense/kernel*
_output_shapes

: 
Ћ
dense/kernelVarHandleOp*
shape
: *
dtype0*
_output_shapes
: *
shared_namedense/kernel*
_class
loc:@dense/kernel
i
-dense/kernel/IsInitialized/VarIsInitializedOpVarIsInitializedOpdense/kernel*
_output_shapes
: 
ї
dense/kernel/AssignAssignVariableOpdense/kernel'dense/kernel/Initializer/random_uniform*
_class
loc:@dense/kernel*
dtype0
ј
 dense/kernel/Read/ReadVariableOpReadVariableOpdense/kernel*
_class
loc:@dense/kernel*
dtype0*
_output_shapes

: 
ї
dense/bias/Initializer/zerosConst*
_class
loc:@dense/bias*
valueB 2        *
dtype0*
_output_shapes
: 
І

dense/biasVarHandleOp*
shape: *
dtype0*
_output_shapes
: *
shared_name
dense/bias*
_class
loc:@dense/bias
e
+dense/bias/IsInitialized/VarIsInitializedOpVarIsInitializedOp
dense/bias*
_output_shapes
: 
{
dense/bias/AssignAssignVariableOp
dense/biasdense/bias/Initializer/zeros*
dtype0*
_class
loc:@dense/bias
ё
dense/bias/Read/ReadVariableOpReadVariableOp
dense/bias*
_class
loc:@dense/bias*
dtype0*
_output_shapes
: 
h
dense/MatMul/ReadVariableOpReadVariableOpdense/kernel*
dtype0*
_output_shapes

: 
h
dense/MatMulMatMulXdense/MatMul/ReadVariableOp*
T0*'
_output_shapes
:          
c
dense/BiasAdd/ReadVariableOpReadVariableOp
dense/bias*
dtype0*
_output_shapes
: 
v
dense/BiasAddBiasAdddense/MatMuldense/BiasAdd/ReadVariableOp*'
_output_shapes
:          *
T0
S

dense/ReluReludense/BiasAdd*'
_output_shapes
:          *
T0
Б
/dense_1/kernel/Initializer/random_uniform/shapeConst*!
_class
loc:@dense_1/kernel*
valueB"       *
dtype0*
_output_shapes
:
Ў
-dense_1/kernel/Initializer/random_uniform/minConst*!
_class
loc:@dense_1/kernel*
valueB 2═;fъао┐*
dtype0*
_output_shapes
: 
Ў
-dense_1/kernel/Initializer/random_uniform/maxConst*!
_class
loc:@dense_1/kernel*
valueB 2═;fъао?*
dtype0*
_output_shapes
: 
м
7dense_1/kernel/Initializer/random_uniform/RandomUniformRandomUniform/dense_1/kernel/Initializer/random_uniform/shape*
T0*!
_class
loc:@dense_1/kernel*
dtype0*
_output_shapes

: 
о
-dense_1/kernel/Initializer/random_uniform/subSub-dense_1/kernel/Initializer/random_uniform/max-dense_1/kernel/Initializer/random_uniform/min*
T0*!
_class
loc:@dense_1/kernel*
_output_shapes
: 
У
-dense_1/kernel/Initializer/random_uniform/mulMul7dense_1/kernel/Initializer/random_uniform/RandomUniform-dense_1/kernel/Initializer/random_uniform/sub*
T0*!
_class
loc:@dense_1/kernel*
_output_shapes

: 
┌
)dense_1/kernel/Initializer/random_uniformAdd-dense_1/kernel/Initializer/random_uniform/mul-dense_1/kernel/Initializer/random_uniform/min*
T0*!
_class
loc:@dense_1/kernel*
_output_shapes

: 
Џ
dense_1/kernelVarHandleOp*
shape
: *
dtype0*
_output_shapes
: *
shared_namedense_1/kernel*!
_class
loc:@dense_1/kernel
m
/dense_1/kernel/IsInitialized/VarIsInitializedOpVarIsInitializedOpdense_1/kernel*
_output_shapes
: 
ћ
dense_1/kernel/AssignAssignVariableOpdense_1/kernel)dense_1/kernel/Initializer/random_uniform*!
_class
loc:@dense_1/kernel*
dtype0
ћ
"dense_1/kernel/Read/ReadVariableOpReadVariableOpdense_1/kernel*!
_class
loc:@dense_1/kernel*
dtype0*
_output_shapes

: 
љ
dense_1/bias/Initializer/zerosConst*
dtype0*
_output_shapes
:*
_class
loc:@dense_1/bias*
valueB2        
Љ
dense_1/biasVarHandleOp*
shape:*
dtype0*
_output_shapes
: *
shared_namedense_1/bias*
_class
loc:@dense_1/bias
i
-dense_1/bias/IsInitialized/VarIsInitializedOpVarIsInitializedOpdense_1/bias*
_output_shapes
: 
Ѓ
dense_1/bias/AssignAssignVariableOpdense_1/biasdense_1/bias/Initializer/zeros*
_class
loc:@dense_1/bias*
dtype0
і
 dense_1/bias/Read/ReadVariableOpReadVariableOpdense_1/bias*
_class
loc:@dense_1/bias*
dtype0*
_output_shapes
:
l
dense_1/MatMul/ReadVariableOpReadVariableOpdense_1/kernel*
dtype0*
_output_shapes

: 
u
dense_1/MatMulMatMul
dense/Reludense_1/MatMul/ReadVariableOp*
T0*'
_output_shapes
:         
g
dense_1/BiasAdd/ReadVariableOpReadVariableOpdense_1/bias*
dtype0*
_output_shapes
:
|
dense_1/BiasAddBiasAdddense_1/MatMuldense_1/BiasAdd/ReadVariableOp*
T0*'
_output_shapes
:         
W
dense_1/ReluReludense_1/BiasAdd*
T0*'
_output_shapes
:         
Б
/dense_2/kernel/Initializer/random_uniform/shapeConst*
dtype0*
_output_shapes
:*!
_class
loc:@dense_2/kernel*
valueB"      
Ў
-dense_2/kernel/Initializer/random_uniform/minConst*!
_class
loc:@dense_2/kernel*
valueB 2      Я┐*
dtype0*
_output_shapes
: 
Ў
-dense_2/kernel/Initializer/random_uniform/maxConst*!
_class
loc:@dense_2/kernel*
valueB 2      Я?*
dtype0*
_output_shapes
: 
м
7dense_2/kernel/Initializer/random_uniform/RandomUniformRandomUniform/dense_2/kernel/Initializer/random_uniform/shape*
T0*!
_class
loc:@dense_2/kernel*
dtype0*
_output_shapes

:
о
-dense_2/kernel/Initializer/random_uniform/subSub-dense_2/kernel/Initializer/random_uniform/max-dense_2/kernel/Initializer/random_uniform/min*
T0*!
_class
loc:@dense_2/kernel*
_output_shapes
: 
У
-dense_2/kernel/Initializer/random_uniform/mulMul7dense_2/kernel/Initializer/random_uniform/RandomUniform-dense_2/kernel/Initializer/random_uniform/sub*
T0*!
_class
loc:@dense_2/kernel*
_output_shapes

:
┌
)dense_2/kernel/Initializer/random_uniformAdd-dense_2/kernel/Initializer/random_uniform/mul-dense_2/kernel/Initializer/random_uniform/min*
T0*!
_class
loc:@dense_2/kernel*
_output_shapes

:
Џ
dense_2/kernelVarHandleOp*
shape
:*
dtype0*
_output_shapes
: *
shared_namedense_2/kernel*!
_class
loc:@dense_2/kernel
m
/dense_2/kernel/IsInitialized/VarIsInitializedOpVarIsInitializedOpdense_2/kernel*
_output_shapes
: 
ћ
dense_2/kernel/AssignAssignVariableOpdense_2/kernel)dense_2/kernel/Initializer/random_uniform*
dtype0*!
_class
loc:@dense_2/kernel
ћ
"dense_2/kernel/Read/ReadVariableOpReadVariableOpdense_2/kernel*!
_class
loc:@dense_2/kernel*
dtype0*
_output_shapes

:
љ
dense_2/bias/Initializer/zerosConst*
_class
loc:@dense_2/bias*
valueB2        *
dtype0*
_output_shapes
:
Љ
dense_2/biasVarHandleOp*
shape:*
dtype0*
_output_shapes
: *
shared_namedense_2/bias*
_class
loc:@dense_2/bias
i
-dense_2/bias/IsInitialized/VarIsInitializedOpVarIsInitializedOpdense_2/bias*
_output_shapes
: 
Ѓ
dense_2/bias/AssignAssignVariableOpdense_2/biasdense_2/bias/Initializer/zeros*
_class
loc:@dense_2/bias*
dtype0
і
 dense_2/bias/Read/ReadVariableOpReadVariableOpdense_2/bias*
_class
loc:@dense_2/bias*
dtype0*
_output_shapes
:
l
dense_2/MatMul/ReadVariableOpReadVariableOpdense_2/kernel*
dtype0*
_output_shapes

:
w
dense_2/MatMulMatMuldense_1/Reludense_2/MatMul/ReadVariableOp*
T0*'
_output_shapes
:         
g
dense_2/BiasAdd/ReadVariableOpReadVariableOpdense_2/bias*
dtype0*
_output_shapes
:
|
dense_2/BiasAddBiasAdddense_2/MatMuldense_2/BiasAdd/ReadVariableOp*
T0*'
_output_shapes
:         
W
dense_2/ReluReludense_2/BiasAdd*
T0*'
_output_shapes
:         
Б
/dense_3/kernel/Initializer/random_uniform/shapeConst*!
_class
loc:@dense_3/kernel*
valueB"      *
dtype0*
_output_shapes
:
Ў
-dense_3/kernel/Initializer/random_uniform/minConst*!
_class
loc:@dense_3/kernel*
valueB 2┤!f1бу┐*
dtype0*
_output_shapes
: 
Ў
-dense_3/kernel/Initializer/random_uniform/maxConst*!
_class
loc:@dense_3/kernel*
valueB 2┤!f1бу?*
dtype0*
_output_shapes
: 
м
7dense_3/kernel/Initializer/random_uniform/RandomUniformRandomUniform/dense_3/kernel/Initializer/random_uniform/shape*
T0*!
_class
loc:@dense_3/kernel*
dtype0*
_output_shapes

:
о
-dense_3/kernel/Initializer/random_uniform/subSub-dense_3/kernel/Initializer/random_uniform/max-dense_3/kernel/Initializer/random_uniform/min*
T0*!
_class
loc:@dense_3/kernel*
_output_shapes
: 
У
-dense_3/kernel/Initializer/random_uniform/mulMul7dense_3/kernel/Initializer/random_uniform/RandomUniform-dense_3/kernel/Initializer/random_uniform/sub*
T0*!
_class
loc:@dense_3/kernel*
_output_shapes

:
┌
)dense_3/kernel/Initializer/random_uniformAdd-dense_3/kernel/Initializer/random_uniform/mul-dense_3/kernel/Initializer/random_uniform/min*
T0*!
_class
loc:@dense_3/kernel*
_output_shapes

:
Џ
dense_3/kernelVarHandleOp*
shared_namedense_3/kernel*!
_class
loc:@dense_3/kernel*
shape
:*
dtype0*
_output_shapes
: 
m
/dense_3/kernel/IsInitialized/VarIsInitializedOpVarIsInitializedOpdense_3/kernel*
_output_shapes
: 
ћ
dense_3/kernel/AssignAssignVariableOpdense_3/kernel)dense_3/kernel/Initializer/random_uniform*!
_class
loc:@dense_3/kernel*
dtype0
ћ
"dense_3/kernel/Read/ReadVariableOpReadVariableOpdense_3/kernel*!
_class
loc:@dense_3/kernel*
dtype0*
_output_shapes

:
љ
dense_3/bias/Initializer/zerosConst*
_class
loc:@dense_3/bias*
valueB2        *
dtype0*
_output_shapes
:
Љ
dense_3/biasVarHandleOp*
shared_namedense_3/bias*
_class
loc:@dense_3/bias*
shape:*
dtype0*
_output_shapes
: 
i
-dense_3/bias/IsInitialized/VarIsInitializedOpVarIsInitializedOpdense_3/bias*
_output_shapes
: 
Ѓ
dense_3/bias/AssignAssignVariableOpdense_3/biasdense_3/bias/Initializer/zeros*
_class
loc:@dense_3/bias*
dtype0
і
 dense_3/bias/Read/ReadVariableOpReadVariableOpdense_3/bias*
_class
loc:@dense_3/bias*
dtype0*
_output_shapes
:
l
dense_3/MatMul/ReadVariableOpReadVariableOpdense_3/kernel*
dtype0*
_output_shapes

:
w
dense_3/MatMulMatMuldense_2/Reludense_3/MatMul/ReadVariableOp*
T0*'
_output_shapes
:         
g
dense_3/BiasAdd/ReadVariableOpReadVariableOpdense_3/bias*
dtype0*
_output_shapes
:
|
dense_3/BiasAddBiasAdddense_3/MatMuldense_3/BiasAdd/ReadVariableOp*
T0*'
_output_shapes
:         
Б
/dense_4/kernel/Initializer/random_uniform/shapeConst*
dtype0*
_output_shapes
:*!
_class
loc:@dense_4/kernel*
valueB"      
Ў
-dense_4/kernel/Initializer/random_uniform/minConst*!
_class
loc:@dense_4/kernel*
valueB 2┤!f1бу┐*
dtype0*
_output_shapes
: 
Ў
-dense_4/kernel/Initializer/random_uniform/maxConst*!
_class
loc:@dense_4/kernel*
valueB 2┤!f1бу?*
dtype0*
_output_shapes
: 
м
7dense_4/kernel/Initializer/random_uniform/RandomUniformRandomUniform/dense_4/kernel/Initializer/random_uniform/shape*
T0*!
_class
loc:@dense_4/kernel*
dtype0*
_output_shapes

:
о
-dense_4/kernel/Initializer/random_uniform/subSub-dense_4/kernel/Initializer/random_uniform/max-dense_4/kernel/Initializer/random_uniform/min*
T0*!
_class
loc:@dense_4/kernel*
_output_shapes
: 
У
-dense_4/kernel/Initializer/random_uniform/mulMul7dense_4/kernel/Initializer/random_uniform/RandomUniform-dense_4/kernel/Initializer/random_uniform/sub*
T0*!
_class
loc:@dense_4/kernel*
_output_shapes

:
┌
)dense_4/kernel/Initializer/random_uniformAdd-dense_4/kernel/Initializer/random_uniform/mul-dense_4/kernel/Initializer/random_uniform/min*
T0*!
_class
loc:@dense_4/kernel*
_output_shapes

:
Џ
dense_4/kernelVarHandleOp*!
_class
loc:@dense_4/kernel*
shape
:*
dtype0*
_output_shapes
: *
shared_namedense_4/kernel
m
/dense_4/kernel/IsInitialized/VarIsInitializedOpVarIsInitializedOpdense_4/kernel*
_output_shapes
: 
ћ
dense_4/kernel/AssignAssignVariableOpdense_4/kernel)dense_4/kernel/Initializer/random_uniform*!
_class
loc:@dense_4/kernel*
dtype0
ћ
"dense_4/kernel/Read/ReadVariableOpReadVariableOpdense_4/kernel*!
_class
loc:@dense_4/kernel*
dtype0*
_output_shapes

:
љ
dense_4/bias/Initializer/zerosConst*
dtype0*
_output_shapes
:*
_class
loc:@dense_4/bias*
valueB2        
Љ
dense_4/biasVarHandleOp*
dtype0*
_output_shapes
: *
shared_namedense_4/bias*
_class
loc:@dense_4/bias*
shape:
i
-dense_4/bias/IsInitialized/VarIsInitializedOpVarIsInitializedOpdense_4/bias*
_output_shapes
: 
Ѓ
dense_4/bias/AssignAssignVariableOpdense_4/biasdense_4/bias/Initializer/zeros*
_class
loc:@dense_4/bias*
dtype0
і
 dense_4/bias/Read/ReadVariableOpReadVariableOpdense_4/bias*
_class
loc:@dense_4/bias*
dtype0*
_output_shapes
:
l
dense_4/MatMul/ReadVariableOpReadVariableOpdense_4/kernel*
dtype0*
_output_shapes

:
w
dense_4/MatMulMatMuldense_2/Reludense_4/MatMul/ReadVariableOp*
T0*'
_output_shapes
:         
g
dense_4/BiasAdd/ReadVariableOpReadVariableOpdense_4/bias*
dtype0*
_output_shapes
:
|
dense_4/BiasAddBiasAdddense_4/MatMuldense_4/BiasAdd/ReadVariableOp*
T0*'
_output_shapes
:         
U
dense_4/ExpExpdense_4/BiasAdd*
T0*'
_output_shapes
:         
Б
/dense_5/kernel/Initializer/random_uniform/shapeConst*!
_class
loc:@dense_5/kernel*
valueB"      *
dtype0*
_output_shapes
:
Ў
-dense_5/kernel/Initializer/random_uniform/minConst*!
_class
loc:@dense_5/kernel*
valueB 2┤!f1бу┐*
dtype0*
_output_shapes
: 
Ў
-dense_5/kernel/Initializer/random_uniform/maxConst*!
_class
loc:@dense_5/kernel*
valueB 2┤!f1бу?*
dtype0*
_output_shapes
: 
м
7dense_5/kernel/Initializer/random_uniform/RandomUniformRandomUniform/dense_5/kernel/Initializer/random_uniform/shape*
T0*!
_class
loc:@dense_5/kernel*
dtype0*
_output_shapes

:
о
-dense_5/kernel/Initializer/random_uniform/subSub-dense_5/kernel/Initializer/random_uniform/max-dense_5/kernel/Initializer/random_uniform/min*
T0*!
_class
loc:@dense_5/kernel*
_output_shapes
: 
У
-dense_5/kernel/Initializer/random_uniform/mulMul7dense_5/kernel/Initializer/random_uniform/RandomUniform-dense_5/kernel/Initializer/random_uniform/sub*
T0*!
_class
loc:@dense_5/kernel*
_output_shapes

:
┌
)dense_5/kernel/Initializer/random_uniformAdd-dense_5/kernel/Initializer/random_uniform/mul-dense_5/kernel/Initializer/random_uniform/min*
T0*!
_class
loc:@dense_5/kernel*
_output_shapes

:
Џ
dense_5/kernelVarHandleOp*
dtype0*
_output_shapes
: *
shared_namedense_5/kernel*!
_class
loc:@dense_5/kernel*
shape
:
m
/dense_5/kernel/IsInitialized/VarIsInitializedOpVarIsInitializedOpdense_5/kernel*
_output_shapes
: 
ћ
dense_5/kernel/AssignAssignVariableOpdense_5/kernel)dense_5/kernel/Initializer/random_uniform*!
_class
loc:@dense_5/kernel*
dtype0
ћ
"dense_5/kernel/Read/ReadVariableOpReadVariableOpdense_5/kernel*!
_class
loc:@dense_5/kernel*
dtype0*
_output_shapes

:
љ
dense_5/bias/Initializer/zerosConst*
dtype0*
_output_shapes
:*
_class
loc:@dense_5/bias*
valueB2        
Љ
dense_5/biasVarHandleOp*
dtype0*
_output_shapes
: *
shared_namedense_5/bias*
_class
loc:@dense_5/bias*
shape:
i
-dense_5/bias/IsInitialized/VarIsInitializedOpVarIsInitializedOpdense_5/bias*
_output_shapes
: 
Ѓ
dense_5/bias/AssignAssignVariableOpdense_5/biasdense_5/bias/Initializer/zeros*
dtype0*
_class
loc:@dense_5/bias
і
 dense_5/bias/Read/ReadVariableOpReadVariableOpdense_5/bias*
dtype0*
_output_shapes
:*
_class
loc:@dense_5/bias
l
dense_5/MatMul/ReadVariableOpReadVariableOpdense_5/kernel*
dtype0*
_output_shapes

:
w
dense_5/MatMulMatMuldense_2/Reludense_5/MatMul/ReadVariableOp*
T0*'
_output_shapes
:         
g
dense_5/BiasAdd/ReadVariableOpReadVariableOpdense_5/bias*
dtype0*
_output_shapes
:
|
dense_5/BiasAddBiasAdddense_5/MatMuldense_5/BiasAdd/ReadVariableOp*
T0*'
_output_shapes
:         ""┬
trainable_variablesфД
x
dense/kernel:0dense/kernel/Assign"dense/kernel/Read/ReadVariableOp:0(2)dense/kernel/Initializer/random_uniform:08
g
dense/bias:0dense/bias/Assign dense/bias/Read/ReadVariableOp:0(2dense/bias/Initializer/zeros:08
ђ
dense_1/kernel:0dense_1/kernel/Assign$dense_1/kernel/Read/ReadVariableOp:0(2+dense_1/kernel/Initializer/random_uniform:08
o
dense_1/bias:0dense_1/bias/Assign"dense_1/bias/Read/ReadVariableOp:0(2 dense_1/bias/Initializer/zeros:08
ђ
dense_2/kernel:0dense_2/kernel/Assign$dense_2/kernel/Read/ReadVariableOp:0(2+dense_2/kernel/Initializer/random_uniform:08
o
dense_2/bias:0dense_2/bias/Assign"dense_2/bias/Read/ReadVariableOp:0(2 dense_2/bias/Initializer/zeros:08
ђ
dense_3/kernel:0dense_3/kernel/Assign$dense_3/kernel/Read/ReadVariableOp:0(2+dense_3/kernel/Initializer/random_uniform:08
o
dense_3/bias:0dense_3/bias/Assign"dense_3/bias/Read/ReadVariableOp:0(2 dense_3/bias/Initializer/zeros:08
ђ
dense_4/kernel:0dense_4/kernel/Assign$dense_4/kernel/Read/ReadVariableOp:0(2+dense_4/kernel/Initializer/random_uniform:08
o
dense_4/bias:0dense_4/bias/Assign"dense_4/bias/Read/ReadVariableOp:0(2 dense_4/bias/Initializer/zeros:08
ђ
dense_5/kernel:0dense_5/kernel/Assign$dense_5/kernel/Read/ReadVariableOp:0(2+dense_5/kernel/Initializer/random_uniform:08
o
dense_5/bias:0dense_5/bias/Assign"dense_5/bias/Read/ReadVariableOp:0(2 dense_5/bias/Initializer/zeros:08"И
	variablesфД
x
dense/kernel:0dense/kernel/Assign"dense/kernel/Read/ReadVariableOp:0(2)dense/kernel/Initializer/random_uniform:08
g
dense/bias:0dense/bias/Assign dense/bias/Read/ReadVariableOp:0(2dense/bias/Initializer/zeros:08
ђ
dense_1/kernel:0dense_1/kernel/Assign$dense_1/kernel/Read/ReadVariableOp:0(2+dense_1/kernel/Initializer/random_uniform:08
o
dense_1/bias:0dense_1/bias/Assign"dense_1/bias/Read/ReadVariableOp:0(2 dense_1/bias/Initializer/zeros:08
ђ
dense_2/kernel:0dense_2/kernel/Assign$dense_2/kernel/Read/ReadVariableOp:0(2+dense_2/kernel/Initializer/random_uniform:08
o
dense_2/bias:0dense_2/bias/Assign"dense_2/bias/Read/ReadVariableOp:0(2 dense_2/bias/Initializer/zeros:08
ђ
dense_3/kernel:0dense_3/kernel/Assign$dense_3/kernel/Read/ReadVariableOp:0(2+dense_3/kernel/Initializer/random_uniform:08
o
dense_3/bias:0dense_3/bias/Assign"dense_3/bias/Read/ReadVariableOp:0(2 dense_3/bias/Initializer/zeros:08
ђ
dense_4/kernel:0dense_4/kernel/Assign$dense_4/kernel/Read/ReadVariableOp:0(2+dense_4/kernel/Initializer/random_uniform:08
o
dense_4/bias:0dense_4/bias/Assign"dense_4/bias/Read/ReadVariableOp:0(2 dense_4/bias/Initializer/zeros:08
ђ
dense_5/kernel:0dense_5/kernel/Assign$dense_5/kernel/Read/ReadVariableOp:0(2+dense_5/kernel/Initializer/random_uniform:08
o
dense_5/bias:0dense_5/bias/Assign"dense_5/bias/Read/ReadVariableOp:0(2 dense_5/bias/Initializer/zeros:08*╔
defaultй
%
default
X:0         0
locs(
dense_3/BiasAdd:0         2
logits(
dense_5/BiasAdd:0         .
scales$
dense_4/Exp:0         