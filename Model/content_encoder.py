# Content encoder 

class UnifontTextEncoder(nn.Module):
    def __init__(self, font_path="/home/kishan/diffusion/NotoSansDevanagari-Regular.ttf", image_size=(64, 1024)):
        super().__init__()
        self.font_path = font_path
        self.image_size = image_size
        self.transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize([0.5] * 3, [0.5] * 3),
        ])
        try:
            self.font = ImageFont.truetype(font_path, 28)
        except Exception:
            self.font = ImageFont.load_default()

    def render_text(self, text):
        H, W = self.image_size
        img = Image.new('RGB', (W, H), (255, 255, 255))
        draw = ImageDraw.Draw(img)

        font_size = 28
        margin = 20
        min_font_size = 8

        # shrink font until text fits within canvas width
        font = self._load_font(font_size)
        while font_size > min_font_size:
            bbox = draw.textbbox((0, 0), text, font=font)
            text_w = bbox[2] - bbox[0]
            if text_w <= (W - margin):
                break
            font_size -= 2
            font = self._load_font(font_size)

        # vertical centering using actual glyph bbox (handles matras above
        # and subscript conjuncts/vowel signs below baseline correctly)
        bbox = draw.textbbox((0, 0), text, font=font)
        text_h = bbox[3] - bbox[1]
        text_w = bbox[2] - bbox[0]

        x = max(0, (W - text_w) // 2 - bbox[0])
        y = max(0, (H - text_h) // 2 - bbox[1])

        draw.text((x, y), text, font=font, fill=(0, 0, 0))
        return img

    def _load_font(self, font_size):
        try:
            # layout_engine=RAQM enables proper Indic shaping (reordering
            # pre-base matras, forming conjuncts) — requires libraqm installed.
            # Without it, Devanagari text renders in raw logical codepoint
            # order, which is visually WRONG for many words.
            return ImageFont.truetype(
                self.font_path,
                font_size,
                layout_engine=ImageFont.Layout.RAQM,
            )
        except Exception:
            try:
                return ImageFont.truetype(self.font_path, font_size)
            except Exception:
                return self.font
    
    def forward(self, texts, device=None):
        imgs = [self.transform(self.render_text(t)) for t in texts]
        imgs = torch.stack(imgs)
        if device is not None:
            imgs = imgs.to(device)
        return imgs

class ContentEncoder(nn.Module):
    def __init__(self, font_path="/home/kishan/diffusion/NotoSansDevanagari-Regular.ttf"):
        super().__init__()
        self.unifont = UnifontTextEncoder(font_path)
        mnet = mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT)
        self.backbone = mnet.features[:7]
        self.proj = nn.Conv2d(32, 64, kernel_size=1)
        # CHANGED: Use pure self-attention so nn.Sequential doesn't crash
        self.attn_head = nn.Sequential(
            SelfAttentionBlock(64), 
            SelfAttentionBlock(64), 
            SelfAttentionBlock(64),
            Resblock(64, 64),
        )
        self.pos_embed = nn.Parameter(torch.zeros(1, 64, 1, 128))  # match feat map W=128
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, text):
        dev = next(self.parameters()).device
        x = self.unifont(text, device=dev)
        x = self.backbone(x)
        x = self.proj(x)
        x = x + self.pos_embed
        Q = self.attn_head(x)
        return Q


