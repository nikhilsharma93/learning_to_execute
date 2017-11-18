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
local backwardConnect = t.backwardConnect


-- Log results to files
local trainLogger = optim.Logger(paths.concat(opt.save, 'trainV2.log'))


----------------------------------------------------------------------
print(sys.COLORS.red ..  '==> flattening model parameters')



----------------------------------------------------------------------
print(sys.COLORS.red ..  '==> configuring optimizer')

local optimState = {
   learningRate = opt.learningRate,
   momentum = opt.momentum,
   weightDecay = opt.weightDecay,
   learningRateDecay = opt.learningRateDecay
}


----------------------------------------------------------------------
print(sys.COLORS.red ..  '==> defining training procedure')

local function loadBatch(start_idx, stop_idx, shuffle, batch_size)
    local idx = 1
    local enc_seq_len = 20
    local dec_seq_len = 20
    local enc_inp = torch.zeros(batch_size, enc_seq_len)
    local dec_inp = torch.zeros(batch_size, dec_seq_len)
    local tar = torch.zeros(batch_size, dec_seq_len)

    for i = start_idx, stop_idx do
        local sample = train_data.encoder_in[shuffle[i]]
        local len = sample:size(1)
        enc_inp[{idx,{enc_seq_len-len+1,enc_seq_len}}] = sample:clone()

        local sample = train_data.decoder_in[shuffle[i]]
        local len = sample:size(1)
        dec_inp[{idx,{1,len}}] = sample:clone()

        local sample = train_data.targets[shuffle[i]]
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
    local filename = 'results/output_text2.txt'
    --print (opt.batchSize)
    for loop_samples = 1,opt.batchSize do
        --print (tar[loop_samples])
        local inp = removeZeros(enc_inp[loop_samples])
        local out = dec_out[loop_samples]
        local target = removeZeros(tar[loop_samples])
        local dict = train_data.reverse_vocab_table

        for i = 1,inp:size(1) do
            out_str = out_str..dict[inp[i]]
        end
        out_str = out_str.."         "

        for i = 1,target:size(1) do
            out_str = out_str..dict[target[i]]
        end
        out_str = out_str.."         "

        --print (out[1])
        --print (out[2])
        --print ('\n')
        for i = 1,out:size(1) do
            local val
            local ind
            val, ind = torch.max(out[i],1)
            --val,ind=torch.sort(out[i])
            --ind = torch.multinomial(out[i]:exp(),1)
            local c = dict[ind[1]]
            if c == train_data.EOS then
                out_str = out_str..c
                break
            end
            out_str = out_str..c
        end
        out_str = out_str.."\n"
    end
    out_str = out_str.."\n\n"
    local file = assert(io.open(filename,"a"))
    file:write(out_str)
    file:close()
    --io.close()
end



local epoch

local function train(trainSet)

    encoder:training()
    decoder:training()

    -- epoch tracker
    epoch = epoch or 1

    -- local vars
    local time = sys.clock()
    local nll = 0
    local batchEpochCount = 0
    -- shuffle at each epoch
    local shuffle = torch.randperm(train_data.size())

    -- Small snippet to be able to change the learning rate at will
    if epoch % 1 == 0 then
        print (sys.COLORS.blue..'Change Learning Rate?')
        local handle = io.popen("bash readEpochChange.sh")
        local content = handle:read("*a")
        handle:close()
        if string.sub(content,1,3) == "yes" then
            print (sys.COLORS.blue .. 'Chaning Learning Rate')
            optimState['learningRate'] = optimState['learningRate']/10.0
        else
            print (sys.COLORS.blue .. 'Learning Rate Unchanged')
        end
    end


    -- do one epoch
    print(sys.COLORS.green .. '\n==> doing epoch on training data:')
    print ('Learning Rate for Epoch '..tostring(epoch)..': '..tostring(optimState['learningRate']))
    print("==> online epoch # " .. epoch .. ' [batchSize = ' .. opt.batchSize .. ']')

    for t = 1,train_data.size(),opt.batchSize do

        -- disp progress
        xlua.progress(t, train_data.size())
        collectgarbage()

        -- batch fits?
        if (t + opt.batchSize - 1) > train_data.size() then
            break
        end

        batchEpochCount = batchEpochCount + 1

        -- create mini batch
        enc_inp, dec_inp, tar, current_enc_seq_len, current_dec_seq_len = loadBatch(t, t+opt.batchSize-1, shuffle, opt.batchSize)
        enc_inp = enc_inp:t()
        dec_inp = dec_inp:t()


        encoder:zeroGradParameters()
        decoder:zeroGradParameters()

        encoder:forget()
	    decoder:forget()

            -- evaluate function for complete mini batch
            --print ('encin: '); print (enc_inp:size())
            local enc_out = encoder:forward(enc_inp)
            --print ('encou: '); print(enc_out:size())
            forwardConnect(encoder, decoder, current_enc_seq_len)
            --print ('decin: '); print (dec_inp:size())
            local dec_out = decoder:forward(dec_inp)
            --print ('decou: '); print(dec_out:size())
            --print ('tar: '); print(tar:size())

            --[[
            dec_out_temp = torch.Tensor(current_enc_seq_len, opt.batchSize, 47)

            for il = 1,current_enc_seq_len do
                dec_out_temp[il] = dec_out[il]
            end
            --print (dec_out)
            --print (tar:size())
            --os.exit()
            --print (dec_out_temp:size())

            local E = loss:forward(dec_out_temp, tar)
            ]]
            local E = loss:forward(dec_out, tar)
            nll = nll + E
            print ('\nnll: ', E, torch.max(dec_out), torch.min(dec_out))

            if true then
                writeOutput(enc_inp, tar, dec_out, t)
            end

            -- Backward pass
            -- estimate df/dW
            local dE_dy = loss:backward(dec_out, tar)
            decoder:backward(dec_inp, dE_dy)
            backwardConnect(encoder, decoder, current_dec_seq_len)
            local zero_tensor = torch.Tensor(enc_out):zero()
            encoder:backward(enc_inp, zero_tensor)

            encoder:gradParamClip(2)
            decoder:gradParamClip(2)

               decoder:updateParameters(optimState['learningRate'])
               encoder:updateParameters(optimState['learningRate'])
    end

    -- time taken
    time = sys.clock() - time
    time = time / train_data.size()
    print("\n==> time to learn 1 sample = " .. (time*1000) .. 'ms')
    print ("Average NLL: "..nll/batchEpochCount)

    -- update logger/plot
    trainLogger:add{['Error On Epochs'] = nll/batchEpochCount}



    if false then
      print (sys.COLORS.blue..'Save Model?')
      local handle = io.popen("bash readEpochChange.sh")
      local content = handle:read("*a")
      handle:close()
      if string.sub(content,1,3) == "yes" then
        print (sys.COLORS.blue .. 'Saving Model')
        local filename = 'results/modelV1.t7'
        --os.execute('mkdir -p ' .. sys.dirname(filename))
        print('==> saving model to '..filename)
        model1 = model:clone()
        --netLighter(model1)
        --torch.save(filename, model1)
        torch.save(filename, model1:clearState())
      else
        print (sys.COLORS.blue .. 'Did Not Save The Model')
      end
    end

    -- next epoch
    epoch = epoch + 1
end

-- Export:
return train
