# Machine Translation for Programming Languages
An LSTM based approach to model programming languages. 

The current version learned to convert small Python code snippets to its equivalent in Lua (its human readable format, not the machine one).

Inspired from Learning to Execute (https://arxiv.org/abs/1410.4615), from which the Python data generation script was borrowed. It was modified to generate target as Lua snippets. 


Requirements: 
`Torch`, `Lua`, `nn, rnn (element-research), cunn (optional, CUDA) libraries`



To generate the data, execute

```
th data_generate.lua
```

Note that the Lua syntax given as target is not entirely true. The current version has some workarounds for the syntax of "if-else" statements. For more information, compare the input and target values provided below.
The network was trained with hardness parameters nesting = 2 and length = 5 (look under data_generate.lua and the paper itself for more details).



To train, execute

```
th run.lua -b <batch-size>  -r <initial learning rate>  -o <save directory path>  -m <momentum, if used> -p (if using CUDA)
```



To see some sample tests, execute

```
th testRun.lua -p (if using CUDA)
```





Sample input, output, and targets:

INPUT
```
h=49910;
d=34710
for x in range(20):d+=(h+31879)
print(d)
```

OUTPUT
```
h=49910;
d=34710
for x=1,20 do d=d+(h+31879); end
print(d)	
```

TARGET
```
h=49910;
d=34710
for x=1,20 do d=d+(h+31879); end
print(d)	
```



INPUT
```
g=29330
for x in range(1):g-=1324
print((26032 if g>88371 else 45290))
```

OUTPUT
```
g=29330
for x=1,1 do g=g-1324; end
print((if g>88371 then 26032 else 45290 end))	
```

TARGET
```
g=29330
for x=1,1 do g=g-1324; end
print((if g>88371 then 26032 else 45290 end))	
```



INPUT
```
print((80290 if 98613>75134 else (19*94506)))
```

OUTPUT
```
print((if 98613>75134 then 80290 else (19*94506) end))	
```

TARGET
```
print((if 98613>75134 then 80290 else (19*94506) end))	
```




