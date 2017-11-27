require 'rnn'
--require 'cunn'
require 'lfs'

EOS = "."
GO = "!"
VOCAB = "abcdefghijklmnopqrstuvwxyz <>:.!,;+=-\n*/()1234567890"
VOCABSIZE = #VOCAB

local vocab_table = {}
for loop_char = 1,#VOCAB do
    local c = VOCAB:sub(loop_char,loop_char)
    vocab_table[c] = loop_char
end

local reverse_vocab_table = {}
for loop_char = 1,#VOCAB do
    local c = VOCAB:sub(loop_char,loop_char)
    table.insert(reverse_vocab_table, c)
end

local function convertToInts(seq)
    local seq_int = torch.Tensor(#seq)
    for loop_s = 1,#seq do
        seq_int[loop_s] = vocab_table[seq:sub(loop_s, loop_s)]
    end
    return seq_int
end

local function forwardConnect(encoder, decoder, seq_len)
   for i=1,#encoder.lstm_layers do
       decoder.lstm_layers[i].userPrevOutput = encoder.lstm_layers[i].output[seq_len]
       decoder.lstm_layers[i].userPrevCell = encoder.lstm_layers[i].cell[seq_len]
   end
end

local function maxProb(dec_out)
    local dec_out = dec_out:clone():permute(2,1,3)
    local out_str = "BATCH "..tostring(t).."\n"
    for loop_samples = 1,1 do
        local out = dec_out[loop_samples]
        local dict = reverse_vocab_table

        for i = 1,out:size(1) do
            local val
            local ind
            val, ind = torch.max(out[i],1)
            --val,ind=torch.sort(out[i])
            --ind = torch.multinomial(out[i]:exp(),1)
            local c = dict[ind[1]]
            if c == EOS then
                out_str = out_str..c
                break
            end
            out_str = out_str..c
        end

        out_str = out_str.."\n\n"
    end
    out_str = out_str.."\n\n"
    return out_str
end


local function getOutput(enc_inp)
    --enc_inp is a string

    local current_enc_seq_len = enc_inp:len()
    enc_inp = convertToInts(enc_inp)
    --print ('size bef: '); print(enc_inp:size())
    enc_inp = enc_inp:view(current_enc_seq_len,1)
    --print ('size bef: '); print(enc_inp:size())

    dec_inp = convertToInts(GO)
    dec_inp = dec_inp:view(1,1)

    local enc_out = encoder:forward(enc_inp)
    --print ('encou: '); print(enc_out:size())
    forwardConnect(encoder, decoder, current_enc_seq_len)
    --print ('decin: '); print (dec_inp:size())
    --print (dec_inp)
    local dec_out = decoder:forward(dec_inp)
    --print ('dec out: '); print (dec_out)
    local val, ind = torch.max(dec_out,3)
    --print ("val ind: ", val, ind[{1,1,1}])
    ind = ind[{1,1,1}]
    local current_char = reverse_vocab_table[ind]
    local out_str = current_char
    --print ('out_str... ', out_str)
    loc = 1
    while (current_char ~= EOS) do
        --print ('doing..')
        --local temp_dec_in = target:sub(loc,loc)
        --print (temp_dec_in)
        --temp_dec_in = vocab_table[temp_dec_in]
        --print (temp_dec_in)
        --dec_inp:fill(temp_dec_in)
        dec_inp:fill(ind)
        dec_out = decoder:forward(dec_inp)
        val, ind = torch.max(dec_out,3)
        ind = ind[{1,1,1}]

        current_char = reverse_vocab_table[ind]

        out_str = out_str..current_char
        loc = loc+1
    end
    return out_str
end

local current_dir = lfs.currentdir()
decoder = torch.load(current_dir..'/results_5_2_500k_b128_r007/modelV1_decoder_cpu.t7'):type('torch.DoubleTensor')
encoder = torch.load(current_dir..'/results_5_2_500k_b128_r007/modelV1_encoder_cpu.t7'):type('torch.DoubleTensor')
encoder:evaluate()
decoder:evaluate()

inputs = {}
targets = {}
table.insert(inputs, "for x in range(4):-=18652")
table.insert(inputs, "print((111 if 45<23 else 129))")
--target = "print(10)."

for loop_inp = 1, #inputs do
    local enc_inp = inputs[loop_inp]
    print ('\nINPUT:\n')
    print (enc_inp)
    encoder:remember('both')
    decoder:remember('both')
    local out_str = getOutput(enc_inp)
    print ('OUTPUT:\n')
    print (out_str)
    encoder:forget()
    decoder:forget()
end
