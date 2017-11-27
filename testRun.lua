require 'rnn'
require 'cunn'

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

local function convertToInts(samples)
    --print ('samples: ')
    --print (samples)
    for loop_samples = 1,#samples do
        local s = samples[loop_samples]
        local s_int = torch.Tensor(#s)
        for loop_s = 1,#s do
            s_int[loop_s] = vocab_table[s:sub(loop_s, loop_s)]
        end
        samples[loop_samples] = s_int
    end
    return samples
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


local function getOutput(enc_inp, dec_inp)
    local current_enc_seq_len = enc_inp[1]:len()
    enc_inp = convertToInts(enc_inp)[1]
    --print ('size bef: '); print(enc_inp:size())
    enc_inp = enc_inp:view(1,current_enc_seq_len)
    --print ('size bef: '); print(enc_inp:size())
    enc_inp = enc_inp:t()

    dec_inp = convertToInts(dec_inp)[1]
    dec_inp = dec_inp:view(1,1)
    dec_inp = dec_inp:t()

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
    out_str = reverse_vocab_table[ind]
    print ('out_str... ', out_str)
    loc = 1
    while (out_str ~= EOS) do
        print ('doing..')
        --dec_inp = convertToInts({out_str})[1]
        --dec_inp = dec_inp:view(1,1)
        --dec_inp = dec_inp:t()
        local temp_dec_in = target:sub(loc,loc)
        temp_dec_in = vocab_table[temp_dec_in]
        dec_inp:fill(temp_dec_in)
        --dec_inp:fill(ind)
        dec_out = decoder:forward(dec_inp)
        local val, ind = torch.max(dec_out,3)
        ind = ind[{1,1,1}]
        out_str = reverse_vocab_table[ind]
        print ('out_str... ', out_str)
        loc = loc+1
    end

end

enc_inp = {}
enc_inp[1] = "print(911810)"
dec_inp = {}
dec_inp[1] = GO
target = "print(915810)."
decoder = torch.load('/home/nikhil/myCode/learning/Torch/learning_to_exc/git/learning_to_execute/results_5_2_500k_b128_r007/modelV1_decoder_epoch3.t7'):type('torch.DoubleTensor')
encoder = torch.load('/home/nikhil/myCode/learning/Torch/learning_to_exc/git/learning_to_execute/results_5_2_500k_b128_r007/modelV1_encoder_epoch3.t7'):type('torch.DoubleTensor')
encoder:evaluate()
decoder:evaluate()
--encoder:remember('both')
decoder:remember('both')
getOutput(enc_inp, dec_inp)
