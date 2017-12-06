# Machine Translation for Programming Languages
An LSTM based approach to model programming languages. 

<br/>
The current version learned to convert small Python code snippets to their equivalent in Lua (its human readable format, not the machine one).

Inspired from Learning to Execute (https://arxiv.org/abs/1410.4615), from which the Python data generation script was borrowed. It was modified to generate target as Lua snippets. <br/><br/>


Requirements: 
`Torch`, `Lua`, `nn, nngraph modules for Torch`

<br />
To generate the data, execute

```
th data_generate.lua
```

Note that the Lua syntax given as target is not entirely true. The current version has some workarounds for the syntax of "if-else" statements. For more information, compare the input and target values provided below.
The network was trained with hardness parameters nesting = 2 and length = 5 (look under data_generate.lua and the paper itself for more details).


<br />
To train, execute

```
th run.lua -b <batch-size>  -r <initial learning rate>  -o <save directory path>  -m <momentum, if used> -p (if using CUDA)
```


<br />
To see some sample tests, execute

```
th testRun.lua
```

<br /><br />
Sample input, output, and targets:

INPUT
```
c=(4*514706)
for x in range(15):c+=827640
print(c)
```

OUTPUT
```
c=(4*514706)
for x=1,15 do c=c+827640; end
print(c)	
```

TARGET
```
c=(4*514706)
for x=1,15 do c=c+827640; end
print(c)	
```

<br /><br />
INPUT
```
a=33759;
print(((a+(242716 if 542419<928910 else 463951))*16))
```

OUTPUT
```
a=33759;
print(((a+(if 542419<928010 then 242716 else 463951 end))*16))
```

TARGET
```
a=33759;
print(((a+(if 542419<928910 then 242716 else 463951 end))*16))
```

<br /><br />
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


<br /><br />
INPUT
```
c=590
for x in range(4):c+=586
print((395 if 275>626 else (885+c)))
```

OUTPUT
```
c=590
for x=1,4 do c=c+586; end
print((if 255>626 then 395 else (885+c) end))
```

TARGET
```
c=590
for x=1,4 do c=c+586; end
print((if 275>626 then 395 else (885+c) end))
```



<br /><br />
INPUT
```
i=836448;
e=893113
for x in range(6):e+=(i-(415587-925524))
print((7*e))
```

OUTPUT
```
i=836448;
e=893113
for x=1,6 do e=e+(i-(62295--925564)); end
print((7*e))
```

TARGET
```
i=836448;
e=893113
for x=1,6 do e=e+(i-(415587-925524)); end
print((7*e))
```


<br /><br />
INPUT
```
print(643503) 
```

OUTPUT
```
print(643503) 
```

TARGET
```
print(643503) 
```


<br /><br />
INPUT
```
f=282;
print((4*((5970+(f+2683)) if 6226<2888 else 5854)))
```

OUTPUT
```
f=282;
print((2*(if 6226<2888 then (5950+(f+2820)) else 5854 end)))
```

TARGET
```
f=282;
print((4*(if 6226<2888 then (5970+(f+2683)) else 5854 end)))
```

<br /><br />
INPUT
```
f=86726
for x in range(15):f-=(90217*11)
print(f)
```

OUTPUT
```
f=86726
for x=1,15 do f=f-(90017*11); end
print(f)
```

TARGET
```
f=86726
for x=1,15 do f=f-(90217*11); end
print(f)
```


<br /><br />
INPUT
```
b=6523;
h=(b+4966)
for x in range(7):h-=3951
print((8*(h*10)))
```

OUTPUT
```
b=6523;
h=(b+4966)
for x=1,7 do h=h-3951; end
print((8*(h*10)))
```

TARGET
```
b=6523;
h=(b+4966)
for x=1,7 do h=h-3951; end
print((8*(h*10)))
```


<br /><br />
INPUT
```
print((11*(85924-(4756-99082))))
```

OUTPUT
```
print((11*(85924-(4756-99082))))
```

TARGET
```
print((11*(85924-(4756-99082))))
```


<br /><br />
INPUT
```
b=(16 if 34>57 else 35);
print((b+87))
```

OUTPUT
```
b=(if 34>55 then 16 else 35 end);
print((b+87))
```

TARGET
```
b=(if 34>57 then 16 else 35 end);
print((b+87))
```


<br /><br />
INPUT
```
e=13;
d=3
for x in range(2):d+=(93+(e-(42 if 30<32 else 97)))
print(d)
```

OUTPUT
```
e=13;
d=3
for x=1,2 do d=d+(684-(e+(if 72<32 then 260 else 97 end))); end
print(d)
```

TARGET
```
e=13;
d=3
for x=1,2 do d=d+(93+(e-(if 30<32 then 42 else 97 end))); end
print(d)
```


<br /><br />
INPUT
```
i=((70-71)*6);
print((i-79))	
```

OUTPUT
```
i=((70-713)*6);
print((i-7))
```

TARGET
```
i=((70-71)*6);
print((i-79))	
```
