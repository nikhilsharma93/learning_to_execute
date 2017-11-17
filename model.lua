------------------------------------------------------------------------------
-- __Author__ = Nikhil Sharma
------------------------------------------------------------------------------
require 'rnn'
require 'nn'

local d = require 'data'

-- parameters
local VOCABSIZE = d.VOCABSIZE
local input_dim = 40 -- embeddingSize
local hidden_dim = 128 -- number of hidden cells per layer
local num_layers = 2 -- number of hidden layers


--Helper functions
--[[ Forward coupling: Copy encoder cell and output to decoder LSTM ]]--
local function forwardConnect(encoder, decoder, seq_len)
   for i=1,#encoder.lstm_layers do
         decoder.lstm_layers[i].userPrevOutput = nn.utils.recursiveCopy(decoder.lstm_layers[i].userPrevOutput, encoder.lstm_layers[i].outputs[seq_len])
         decoder.lstm_layers[i].userPrevCell = nn.utils.recursiveCopy(decoder.lstm_layers[i].userPrevCell, encoder.lstm_layers[i].cells[seq_len])
   end
end

--[[ Backward coupling: Copy decoder gradients to encoder LSTM ]]--
local function backwardConnect(encoder, decoder, seq_len)
   for i=1,#encoder.lstm_layers do
         encoder:setGradHiddenState(seq_len, decoder:getGradHiddenState(0))
   end
end



-----------------------------------------------------------
-- Encoder Model
-----------------------------------------------------------
local encoder = nn.Sequential()

-- lookup table
local lookup = nn.LookupTableMaskZero(VOCABSIZE, input_dim)
    encoder:add(lookup)
    encoder:add(nn.SplitTable(1,2)) --split tensors of batch-size into table of batch-size

-- LSTM
encoder.lstm_layers = {}
for i=1,num_layers do
    encoder.lstm_layers[i] = nn.FastLSTM(input_dim, hidden_dim):maskZero(1)
    encoder:add(nn.Sequencer(encoder.lstm_layers[i]))
    input_dim = hidden_dim
end

--Select the output of the last input in the sequence
--encoder:add(nn.Select(1, -1))
encoder:add(nn.SelectTable(-1))



-----------------------------------------------------------
-- Decoder Model
-----------------------------------------------------------
local decoder = nn.Sequential()
--local decoder = nn.Sequential()

-- lookup table
local lookup = nn.LookupTableMaskZero(VOCABSIZE, input_dim)
    decoder:add(lookup)
    decoder:add(nn.SplitTable(1,2)) --split tensors of batch-size into table of batch-size

-- LSTM
decoder.lstm_layers = {}
for i=1,num_layers do
    decoder.lstm_layers[i] = nn.FastLSTM(hidden_dim, hidden_dim):maskZero(1)
    decoder:add(nn.Sequencer(decoder.lstm_layers[i]))
end

decoder:add(nn.Sequencer(nn.MaskZero(nn.Linear(hidden_dim, VOCABSIZE), 1)))
decoder:add(nn.Sequencer(nn.MaskZero(nn.LogSoftMax(), 1)))





-----------------------------------------------------------
-- Loss Function
-----------------------------------------------------------
local criterion = nn.SequencerCriterion(nn.MaskZeroCriterion(nn.ClassNLLCriterion(),1))


-- return package:
return {
   encoder = encoder,
   decoder = decoder,
   loss = criterion,
   forwardConnect = forwardConnect,
   backwardConnect = backwardConnect
}
