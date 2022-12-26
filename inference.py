import argparse
import cv2
from PIL import Image
from models.afs import AFS
import torch

parser = argparse.ArgumentParser()
# pretrained weights
parser.add_argument("--e4e_path", type=str, default="./weights/e4e_ffhq_encode.pt")
parser.add_argument("--style_extraction_path", type=str, default="./weights/style_extraction.pth")
parser.add_argument("--stylegan_path", type=str, default="./weights/stylegan2-ffhq-config-f.pt")
parser.add_argument("--face_parsing_path", type=str, default="./weights/face_parsing.pth")

parser.add_argument("--src_img", type=str, default="demo/source.jpg")
parser.add_argument("--tgt_img", type=str, default="demo/target.jpg")
parser.add_argument("--save_dir", type=str, default="demo/output.jpg")

args = parser.parse_args()


if __name__ == '__main__':
    afs = AFS(ckpt_e4e=args.e4e_path,
              ckpt_style_extraction=args.style_extraction_path,
              ckpt_face_parsing=args.face_parsing_path,
              ckpt_stylegan=args.stylegan_path).cuda().eval()

    src_img = cv2.imread(args.src_img)
    src_img = cv2.cvtColor(src_img, cv2.COLOR_BGR2RGB)
    src_img = ((torch.tensor(src_img) - 127.5) / 127.5).permute(2, 0, 1).unsqueeze(0).cuda()

    tgt_img = cv2.imread(args.tgt_img)
    tgt_img = cv2.cvtColor(tgt_img, cv2.COLOR_BGR2RGB)
    tgt_img = ((torch.tensor(tgt_img) - 127.5) / 127.5).permute(2, 0, 1).unsqueeze(0).cuda()
    with torch.no_grad():
        out_img = afs(src_img, tgt_img)[0] * 127.5 + 127.5

    out_img = out_img.permute(1, 2, 0).cpu().detach().numpy()
    # breakpoint()
    cv2.imwrite(args.save_dir, out_img[:, :, ::-1])
