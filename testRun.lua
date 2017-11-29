------------------------------------------------------------------------------
-- __Author__ = Nikhil Sharma

--Script to run sample test cases with the trained model
--For each input, the expected output and output from model are both shown
--The sample data is generated from the data_generate.lua script
------------------------------------------------------------------------------

require 'rnn'
require 'cunn'
require 'lfs'

--Define some constants. These are in accordance with those defined in data.lua and model.lua
EOS = "."
GO = "!"
VOCAB = "abcdefghijklmnopqrstuvwxyz <>:.!,;+=-\n*/()1234567890"
VOCABSIZE = #VOCAB

--Some helper functions
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


local function getOutput(enc_inp)
    --enc_inp is a string

    local current_enc_seq_len = enc_inp:len()
    enc_inp = convertToInts(enc_inp)
    enc_inp = enc_inp:view(current_enc_seq_len,1)

    --Set the GO indicator for the decoder
    dec_inp = convertToInts(GO)
    dec_inp = dec_inp:view(1,1)

    local enc_out = encoder:forward(enc_inp)

    --Connect the last hidden state of encoder to decoder
    forwardConnect(encoder, decoder, current_enc_seq_len)

    local out_str = ""

    --Keep decoding until EOS character is outputted
    while true do
        local dec_out = decoder:forward(dec_inp)
        local val, ind = torch.max(dec_out,3)
        ind = ind[{1,1,1}]

        local current_char = reverse_vocab_table[ind]
        if (current_char == EOS) then break end
        out_str = out_str..current_char
        dec_inp:fill(ind)
    end
    return out_str
end


local current_dir = lfs.currentdir()

--Load the models
decoder = torch.load(current_dir..'/models/modelV1_decoder.t7'):type('torch.DoubleTensor')
encoder = torch.load(current_dir..'/models/modelV1_encoder.t7'):type('torch.DoubleTensor')
encoder:evaluate()
decoder:evaluate()

--Load sample test data
sample_test_data_dir = current_dir..'/sample_test_data/'
inputs = torch.load(sample_test_data_dir..'input.dat')
targets = torch.load(sample_test_data_dir..'target.dat')

--Loop over the samples and predict!
for loop_inp = 1, #inputs do
    local enc_inp = inputs[loop_inp]
    print ('\nINPUT:')
    print (enc_inp)
    encoder:remember('both')
    decoder:remember('both')
    local out_str = getOutput(enc_inp)
    print ('\nOUTPUT:')
    print (out_str)
    print ('\nEXPECTED OUTPUT:')
    print (targets[loop_inp])
    print ('\n')
    encoder:forget()
    decoder:forget()
end
