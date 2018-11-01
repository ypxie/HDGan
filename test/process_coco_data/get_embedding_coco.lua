-- Modification from the codebase of scott's icml16
-- please check https://github.com/reedscot/icml2016 for details

-- require 'image'
require 'nn'
require 'nngraph'
require 'cunn'
require 'cutorch'
require 'cudnn'
require 'lfs'
require 'torch'
require 'paths'

torch.setdefaulttensortype('torch.FloatTensor')

local alphabet = "abcdefghijklmnopqrstuvwxyz0123456789-,;.!?:'\"/\\|_@#$%^&*~`+-=<>()[]{} "
local dict = {}
for i = 1,#alphabet do
    dict[alphabet:sub(i,i)] = i
end
ivocab = {}
for k,v in pairs(dict) do
  ivocab[v] = k
end

opt = {
  filenames = 'Data/coco/example_captions.t7',
  doc_length = 201,
  queries = 'Data/coco/val/captions.json',
  net_txt = 'Data/coco/coco_gru18_bs64_cls0.5_ngf128_ndf128_a10_c512_80_net_T.t7',
  path = 'Data/coco/val2014_ex_t7/'
}


for k,v in pairs(opt) do opt[k] = tonumber(os.getenv(k)) or os.getenv(k) or opt[k] end
print(opt)

net_txt = torch.load(opt.net_txt)
if net_txt.protos ~=nil then net_txt = net_txt.protos.enc_doc end


net_txt:evaluate()

-- -- Extract all text features.
-- local fea_txt = {}
-- -- Decode text for sanity check.
-- local raw_txt = {}
-- local raw_img = {}
path = 'Data/coco/val2014_ex_t7'
files = paths.dir(path..'/caption_txt')

print (table.getn(files) - 3 + 1)

for i = 3, table.getn(files)  do
  f = files[i]
  if string.find(f, "jpg") then
    tmp_path = path..'/caption_txt/'..f
    
    name = f:sub(1,-5)
        
    c = 1
    feats = torch.Tensor(5, 1024):zero()
    for query_str in io.lines(tmp_path) do
     
      -- print('\t'..query_str)
      
      local txt = torch.zeros(1, opt.doc_length, #alphabet)
      for t = 1,opt.doc_length do
        local ch = query_str:sub(t,t)
        local ix = dict[ch]
        if ix ~= 0 and ix ~= nil then
          txt[{1,t,ix}] = 1
        end
      end
      -- raw_txt[#raw_txt+1] = query_str
      txt = txt:cuda()
      feat = net_txt:forward(txt):clone()
      -- print (feat)
      -- fea_txt[#fea_txt+1] = net_txt:forward(txt):clone()
      -- print (c)
      feats[{{c}}] = feat:float()
      c = c + 1
      if c == 6 then break end
    end

    if c  ~= 6 then
      print ('error')
      break
    end
    torch.save(opt.path..name:sub(1,-5)..'.t7', {txt=feats, img=name})
  end
  if i % 100 == 0 then
    print (i..' processed')
  end
end