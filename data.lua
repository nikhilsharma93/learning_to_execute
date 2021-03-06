------------------------------------------------------------------------------
-- __Author__ = Nikhil Sharma

--Script to load the data, which is generated by the data_generate.lua script
--Returns train and test data
------------------------------------------------------------------------------

require 'lfs'

--Define some constants
EOS = "."
GO = "!"
VOCAB = "abcdefghijklmnopqrstuvwxyz <>:.!,;+=-\n*/()1234567890"
VOCABSIZE = #VOCAB
print("vocab size is: ", VOCABSIZE)

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

function load_data(target_val)
    local number = #target_val
    local decoder_in = {table.unpack(target_val)}
    for loop_samples = 1,number do
        target_val[loop_samples] = target_val[loop_samples]..EOS
        decoder_in[loop_samples] = GO..decoder_in[loop_samples]
    end
    return target_val, decoder_in
end

function convertToInts(samples)
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

training_val_path = lfs.currentdir()..'/data_pyToLua/mix_3to6l_1to4n/training_val.dat'
target_val_path = lfs.currentdir()..'/data_pyToLua/mix_3to6l_1to4n/target_val.dat'

training_val = torch.load(training_val_path)
target_val = torch.load(target_val_path)

samples = training_val
local num_samples = #samples
targets, decoder_in = load_data(target_val)
samples = convertToInts(samples)
targets = convertToInts(targets)
decoder_in = convertToInts(decoder_in)

--80% training, 20% testing
num_train_samples = torch.round(0.8*num_samples)
num_test_samples = num_samples - num_train_samples

train_data = {
    encoder_in = {table.unpack(samples, 1, num_train_samples)},
    decoder_in = {table.unpack(decoder_in, 1, num_train_samples)},
    size = function() return num_train_samples end,
    number = num_train_samples,
    targets = {table.unpack(targets, 1, num_train_samples)},
    reverse_vocab_table = reverse_vocab_table,
    EOS = EOS
}

test_data = {
    encoder_in = {table.unpack(samples, num_train_samples+1,num_samples)},
    decoder_in = {table.unpack(decoder_in, num_train_samples+1,num_samples)},
    size = function() return num_test_samples end,
    number = num_test_samples,
    targets = {table.unpack(targets, num_train_samples+1,num_samples)},
    reverse_vocab_table = reverse_vocab_table,
    EOS = EOS
}



return {
    train_data = train_data,
    test_data = test_data,
    VOCABSIZE = VOCABSIZE,
}
