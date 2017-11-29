------------------------------------------------------------------------------
-- __Author__ = Nikhil Sharma
------------------------------------------------------------------------------

require 'torch'
require 'xlua'
require 'optim'
require 'os'
require 'rnn'
require 'nn'

----------------------------------------------------------------------
-- Model + Loss + Data:
local t = require 'model'
local encoder = t.encoder
local decoder = t.decoder
local loss = t.loss
local forwardConnect = t.forwardConnect


-- Log results to files
local testLogger = optim.Logger(paths.concat(opt.save, 'testV1.log'))


local zero_tensor = torch.Tensor()

if opt.type == 'cuda' then
    print(sys.COLORS.red ..  '==> allocating CUDA memory')
    local enc_inp = torch.CudaTensor()
    local dec_inp = torch.CudaTensor()
    local tar = torch.CudaTensor()
    zero_tensor = torch.CudaTensor()
end


----------------------------------------------------------------------
print(sys.COLORS.red ..  '==> defining testing procedure')

local function loadBatch(start_idx, stop_idx, shuffle, batch_size, reverse_inp, duplicate_inp)
    local reverse_inp = reverse_inp or false
    local duplicate_inp = duplicate_inp or false
    local idx = 1
    local enc_seq_len = 100
    local dec_seq_len = 100
    local enc_inp = torch.zeros(batch_size, enc_seq_len)
    local dec_inp = torch.zeros(batch_size, dec_seq_len)
    local tar = torch.zeros(batch_size, dec_seq_len)

    for i = start_idx, stop_idx do
        local sample = test_data.encoder_in[shuffle[i]]:clone()
        local len = sample:size(1)
        if reverse_inp then sample = sample:index(1 ,torch.linspace(len,1,len):long()) end
        enc_inp[{idx,{enc_seq_len-len+1,enc_seq_len}}] = sample
        if duplicate_inp then enc_inp[{idx,{enc_seq_len-len+1-len,enc_seq_len-len}}] = sample end

        local sample = test_data.decoder_in[shuffle[i]]
        local len = sample:size(1)
        dec_inp[{idx,{1,len}}] = sample:clone()

        local sample = test_data.targets[shuffle[i]]
        local len = sample:size(1)
        tar[{idx,{1, len}}] = sample:clone()

        idx = idx + 1
    end
    return enc_inp, dec_inp, tar:t(), enc_seq_len, dec_seq_len
end

local function removeZeros(inp)
    return inp[inp:ne(0)]
end

local function writeOutput(enc_inp, tar, dec_out, t)
    local enc_inp = enc_inp:clone():permute(2,1)
    local dec_out = dec_out:clone():permute(2,1,3)
    local tar = tar:clone():permute(2,1)
    local out_str = "BATCH "..tostring(t).."\n"
    local filename = paths.concat(opt.save, 'test_output_text1.txt')

    for loop_samples = 1,opt.batchSize do
        local inp = removeZeros(enc_inp[loop_samples])
        local out = dec_out[loop_samples]
        local target = removeZeros(tar[loop_samples])
        local dict = test_data.reverse_vocab_table

        for i = 1,inp:size(1) do
            out_str = out_str..dict[inp[i]]
        end
        out_str = out_str.."         "

        for i = 1,target:size(1) do
            out_str = out_str..dict[target[i]]
        end
        out_str = out_str.."         "

        for i = 1,out:size(1) do
            local val
            local ind
            val, ind = torch.max(out[i],1)
            local c = dict[ind[1]]
            if c == test_data.EOS then
                out_str = out_str..c
                break
            end
            out_str = out_str..c
        end

        out_str = out_str.."\n\n"
    end
    out_str = out_str.."\n\n"
    local file = assert(io.open(filename,"a"))
    file:write(out_str)
    file:close()
end


local epoch

local function test(testSet)

    encoder:evaluate()
    decoder:evaluate()

    -- epoch tracker
    epoch = epoch or 1

    -- local vars
    local time = sys.clock()
    local nll = 0
    local batchEpochCount = 0
    -- shuffle at each epoch
    local shuffle = torch.randperm(test_data.size())


    -- do one epoch
    print(sys.COLORS.green .. '\n==> doing epoch on testing data:')
    print("==> online epoch # " .. epoch .. ' [batchSize = ' .. opt.batchSize .. ']')

    for t = 1,test_data.size(),opt.batchSize do

        -- disp progress
        xlua.progress(t, test_data.size())
        collectgarbage()

        -- batch fits?
        if (t + opt.batchSize - 1) > test_data.size() then
            break
        end

        batchEpochCount = batchEpochCount + 1

        -- create mini batch
        enc_inp, dec_inp, tar, current_enc_seq_len, current_dec_seq_len = loadBatch(t, t+opt.batchSize-1, shuffle, opt.batchSize)
        enc_inp = enc_inp:t()
        dec_inp = dec_inp:t()


        local enc_out = encoder:forward(enc_inp)
        forwardConnect(encoder, decoder, current_enc_seq_len)
        local dec_out = decoder:forward(dec_inp)


        local E = loss:forward(dec_out, tar)
        nll = nll + E
        print ('\nnll: ', E, torch.max(dec_out), torch.min(dec_out))

        if ( (t-1)%(opt.batchSize*4) == 0 ) then
            writeOutput(enc_inp, tar, dec_out, t)
        end

    end

    -- time taken
    time = sys.clock() - time
    time = time / test_data.size()
    print("\n==> time to learn 1 sample = " .. (time*1000) .. 'ms')
    print ("Average NLL: "..nll/batchEpochCount)

    -- update logger
    testLogger:add{['Error On Epochs'] = nll/batchEpochCount}

    -- next epoch
    epoch = epoch + 1
end

-- Export:
return test
