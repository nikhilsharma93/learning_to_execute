------------------------------------------------------------------------------
-- __Author__ = Nikhil Sharma

--Script to run sample test cases with the trained model
--For each input, the expected output and output from model are both shown
--The sample data is generated from the data_generate.lua script
------------------------------------------------------------------------------

require 'rnn'
require 'cunn'
require 'lfs'


torch.setdefaulttensortype('torch.FloatTensor')
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


local function writeError(err, nesting_val, length_val, filename)
    local out_str = "nesting = "..tostring(nesting_val).." length = "..tostring(length_val).." error = "..tostring(err)
    out_str = out_str..'\n'
    local file = assert(io.open(filename,"a"))
    file:write(out_str)
    file:close()
end

local function writeOutput(enc_inp, out, exp_output, files)
    local filename = in_dir..'/outputs/'..files:sub(1,#files-4)..'.txt'
    local out_str = 'INPUT\n'..enc_inp..'\nOUTPUT\n'..out..'\nEXPECTED\n'..exp_output..'\n'
    local file = assert(io.open(filename,"a"))
    file:write(out_str)
    file:close()
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
local error_log = current_dir..'/test_nesting_length/results/error.log'

--Load the models
decoder = torch.load(current_dir..'/models/modelV1_decoder.t7')--:type('torch.DoubleTensor')
encoder = torch.load(current_dir..'/models/modelV1_encoder.t7')--:type('torch.DoubleTensor')
encoder:evaluate()
decoder:evaluate()


function notInTable(files)
    for k,v in ipairs(done) do
        if v==files then return false end
    end
    return true
end

done = {'input_nesting_7_length_7.dat',
	'input_nesting_7_length_1.dat',
	'input_nesting_4_length_10.dat',
	'input_nesting_4_length_9.dat',
	'input_nesting_3_length_10.dat',
	'input_nesting_8_length_1.dat',
	'input_nesting_1_length_10.dat',
	'input_nesting_7_length_9.dat',
	'input_nesting_3_length_9.dat'}

--Load sample test data
in_dir = current_dir..'/test_nesting_length/'
for files in lfs.dir(in_dir) do
    if string.find(files, '.dat') and string.find(files, 'input')  then
        local temp, nesting_end = string.find(files,'nesting_')
        local length_start, length_end = string.find(files,'_length_')
        local nesting_val = files:sub(nesting_end+1,length_start-1)
        local length_val = files:sub(length_end+1, #files-4)

        if ( (tonumber(nesting_val) == 7 )  and tonumber(length_val) == 7) then

            inputs = torch.load(in_dir..files)
            print ('', files, #inputs)
            local gen_file_name = files:sub(7,#files)
            local target_file_name = 'target_'..gen_file_name
            targets = torch.load(in_dir..target_file_name)
            local batch_error = 0

            --Loop over the samples and predict!
            for loop_inp = 1, #inputs do
                local sample_error = 0
                local enc_inp = inputs[loop_inp]
                --enc_inp:cuda()
                --print ('\nINPUT:')
                --print (enc_inp)
                encoder:remember('both')
                decoder:remember('both')
                local out_str = getOutput(enc_inp)
                --print ('\nOUTPUT:')
                --print (out_str)
                --print ('\nEXPECTED OUTPUT:')
                exp_output = targets[loop_inp]
                --print (exp_output)
                --print ('\n')
                encoder:forget()
                decoder:forget()
                writeOutput(enc_inp, out_str, exp_output, files)

                --[[
                local target_length = #exp_output
                local output_length = #out_str
                local min_length = math.min(target_length, output_length)

                --Iterate over strings and find difference
                for loop_str = 1,min_length do
                    local s1 = exp_output:sub(loop_str, loop_str)
                    local s2 = out_str:sub(loop_str, loop_str)
                    if s1 ~= s2 then sample_error = sample_error+1 end
                end

                --account for difference in lengths
                sample_error = sample_error + math.abs(target_length - output_length)
                sample_error = sample_error/target_length
                if (sample_error ~= 0) then
                    print ('\n\nERROR FOUND')
                    print ('inp')
                    print (enc_inp)
                    print ('tar')
                    print (exp_output)
                    print ('out')
                    print (out_str)
                    print ('Error = ', sample_error, math.abs(target_length - output_length))
                end

                --add to the batch_error
                batch_error = batch_error + sample_error
                ]]
            end

            --batch_error
            --batch_error = batch_error/#inputs
            --writeError(batch_error, nesting_val, length_val, error_log)
        end
    end
end
