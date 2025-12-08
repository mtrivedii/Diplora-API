"""Simple LeadAwareResNet1D - Minimal architecture for debugging CPU issues"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def _choose_gn_groups(num_channels: int) -> int:
    """Choose a valid GroupNorm group count that divides num_channels."""
    if num_channels % 8 == 0:
        return 8
    if num_channels % 4 == 0:
        return 4
    if num_channels % 2 == 0:
        return 2
    return 1


class WSConv1d(nn.Conv1d):
    """Conv1d with Weight Standardization.
    Standardizes weights per-outchannel before convolution to stabilize training.
    """
    def forward(self, x):
        weight = self.weight
        weight_mean = weight.mean(dim=(1, 2), keepdim=True)
        weight = weight - weight_mean
        var = weight.var(dim=(1, 2), unbiased=False, keepdim=True)
        weight = weight / torch.sqrt(var + 1e-5)
        return F.conv1d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)


class SEBlock1D(nn.Module):
    """Squeeze-and-Excitation for 1D feature maps."""
    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        reduced = max(1, channels // reduction)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, reduced, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(reduced, channels, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _ = x.shape
        s = self.pool(x).view(b, c)
        s = self.fc(s).view(b, c, 1)
        return x * s


class LeadGating(nn.Module):
    """Lightweight per-lead gating with softmax normalization across leads.
    Operates on concatenated per-lead feature maps [B, (L*C), T] and returns same shape.
    """
    def __init__(self, n_leads: int, channels_per_lead: int, reduction: int = 4, temperature: float = 1.0):
        super().__init__()
        hidden = max(1, channels_per_lead // reduction)
        self.n_leads = int(n_leads)
        self.channels_per_lead = int(channels_per_lead)
        self.temperature = float(temperature)
        self.fc = nn.Sequential(
            nn.Linear(channels_per_lead, hidden, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, 1, bias=True),
        )
        self.pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):
        # x: [B, L*C, T]
        b, lc, t = x.shape
        l = self.n_leads
        c = self.channels_per_lead
        assert lc == l * c, "LeadGating: mismatched concatenated channels"

        # Reshape to [B, L, C, T]
        x4 = x.view(b, l, c, t)
        # Temporal average -> [B, L, C]
        s = self.pool(x4.reshape(b * l, c, t)).view(b, l, c)
        # Per-lead score -> [B, L, 1]
        scores = self.fc(s)
        # Softmax across leads
        weights = torch.softmax(scores / max(1e-6, self.temperature), dim=1)  # [B, L, 1]
        # Apply weights
        x4 = x4 * weights.unsqueeze(-1)
        # Back to [B, L*C, T]
        return x4.view(b, l * c, t)


class RhythmContextHead(nn.Module):
    """Parallel branch for rhythm features (HRV + demographics).
    Processes HRV features (RR intervals, RMSSD, SDNN) and demographic data (age, sex)
    to provide rhythm context for classification.
    """
    def __init__(self, hidden_dim=128):
        super().__init__()
        # Input: 12 HRV features (10 RR + RMSSD + SDNN) + 2 demographics (age, sex) = 14 features
        # Small MLP: 14 -> 64 -> 64 -> hidden_dim
        self.mlp = nn.Sequential(
            nn.Linear(14, 64, bias=True),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(64, 64, bias=True),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(64, hidden_dim, bias=True)
        )
    
    def forward(self, hrv_features, age, sex):
        """
        Args:
            hrv_features: [B, 12] HRV features
            age: [B] or [B, 1] normalized age
            sex: [B] or [B, 1] encoded sex
        Returns:
            [B, hidden_dim] rhythm context embedding
        """
        # Ensure age and sex are [B, 1]
        if age.dim() == 1:
            age = age.unsqueeze(-1)
        if sex.dim() == 1:
            sex = sex.unsqueeze(-1)
        
        # Concatenate all context features
        context = torch.cat([hrv_features, age, sex], dim=1)  # [B, 14]
        return self.mlp(context)  # [B, hidden_dim]


class BlurPool1D(nn.Module):
    """Anti-aliased downsampling with depthwise blur then stride-2 subsample."""
    def __init__(self, channels: int, stride: int = 2, kernel=None):
        super().__init__()
        assert stride in (2, 3), "BlurPool1D supports stride 2 or 3"
        if kernel is None:
            # [1, 2, 1] kernel, normalized
            k = torch.tensor([1., 2., 1.])
        else:
            k = torch.tensor(kernel, dtype=torch.float32)
        k = (k / k.sum()).view(1, 1, -1)
        self.register_buffer('kernel', k)
        self.stride = stride
        self.channels = channels

    def forward(self, x):
        # Depthwise blur
        k = self.kernel.expand(self.channels, 1, -1)
        x = F.conv1d(x, k, stride=1, padding=(k.shape[-1] // 2), groups=self.channels)
        # Subsample
        return x[:, :, :: self.stride]


class SimpleResBlock(nn.Module):
    """Simple residual block with GroupNorm, WS convs, optional SE, no dilations"""
    
    def __init__(self, in_ch, out_ch, stride=1, k=7, use_se=True):
        super().__init__()
        pad = (k - 1) // 2
        
        # Two conv layers with GroupNorm and WS
        self.conv1 = WSConv1d(in_ch, out_ch, k, stride=stride, padding=pad, bias=False)
        self.gn1 = nn.GroupNorm(_choose_gn_groups(out_ch), out_ch)
        self.relu = nn.ReLU(inplace=True)
        
        self.conv2 = WSConv1d(out_ch, out_ch, k, stride=1, padding=pad, bias=False)
        self.gn2 = nn.GroupNorm(_choose_gn_groups(out_ch), out_ch)
        
        # Optional SE
        self.se = SEBlock1D(out_ch, reduction=8) if use_se else nn.Identity()
        
        # Skip connection
        if stride != 1 or in_ch != out_ch:
            self.downsample = nn.Sequential(
                WSConv1d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.GroupNorm(_choose_gn_groups(out_ch), out_ch)
            )
        else:
            self.downsample = nn.Identity()
    
    def forward(self, x):
        identity = self.downsample(x)
        
        out = self.conv1(x)
        out = self.gn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.gn2(out)
        out = self.se(out)
        
        out = out + identity  # Residual connection
        out = self.relu(out)
        
        return out


class LeadAwareResNet1D(nn.Module):
    """
    Simplified Lead-Aware ResNet for ECG Classification
    
    Architecture:
    - Per-lead stem processing (12 separate stems)
    - Simple cross-lead fusion
    - 4-stage ResNet (no SE attention, no dilations)
    - Much simpler for debugging
    
    Removed for debugging:
    - SE attention blocks
    - Dilation patterns
    - Gradient checkpointing
    - Complex fusion
    """
    
    def __init__(self, n_leads=3, num_labels=6, base=16, depths=(2,2,3,2), 
                 k=7, p_drop=0.3, use_lead_mixer: bool = False, lead_mixer_type: str = "gating",
                 use_rhythm_head: bool = False, use_new_arch: bool = False):
        super().__init__()
        
        # Store configuration
        self.n_leads = int(n_leads)
        self.num_labels = int(num_labels)
        self.base = int(base)
        self.depths = tuple(int(d) for d in depths)
        self.k = int(k)
        self.p_drop = float(p_drop)
        self.use_lead_mixer = bool(use_lead_mixer)
        self.lead_mixer_type = str(lead_mixer_type)
        self.use_rhythm_head = bool(use_rhythm_head)
        self.use_new_arch = bool(use_new_arch)
        
        self.config = {
            "arch": "SimpleLeadAwareResNet1D",
            "n_leads": self.n_leads,
            "num_labels": self.num_labels,
            "base": self.base,
            "depths": list(self.depths),
            "k": self.k,
            "p_drop": self.p_drop,
            "use_lead_mixer": self.use_lead_mixer,
            "lead_mixer_type": self.lead_mixer_type,
            "use_rhythm_head": self.use_rhythm_head,
            "use_new_arch": self.use_new_arch,
        }
        
        # Per-lead stem processing (12 separate stems) - KEPT, GN+WS
        self.stems = nn.ModuleList([
            nn.Sequential(
                WSConv1d(1, base, kernel_size=11, stride=2, padding=5, bias=False),
                nn.GroupNorm(_choose_gn_groups(base), base),
                nn.ReLU(inplace=True)
            ) for _ in range(n_leads)
        ])
        # Optional lead mixer operating on concatenated per-lead features
        if self.use_lead_mixer and self.lead_mixer_type == "gating":
            self.lead_mixer = LeadGating(n_leads=n_leads, channels_per_lead=base, reduction=4, temperature=1.0)
        else:
            self.lead_mixer = nn.Identity()
        
        # SIMPLIFIED cross-lead fusion (1x1 conv + GN + ReLU)
        self.fuse = nn.Sequential(
            WSConv1d(n_leads * base, 2 * base, 1, bias=False),
            nn.GroupNorm(_choose_gn_groups(2 * base), 2 * base),
            nn.ReLU(inplace=True)
        )
        
        # ResNet stages with increasing channels (reduced from original)
        C1, C2, C3, C4 = 2*base, 4*base, 8*base, 16*base
        
        # Stage 1: base channels (no downsampling)
        self.layer1 = self._make_stage(C1, C1, depths[0], stride=1, k=k)
        
        # Stage 2: 2× channels, anti-aliased downsample
        self.layer2 = self._make_stage(C1, C2, depths[1], stride=2, k=k)
        
        # Stage 3: 4× channels, anti-aliased downsample  
        self.layer3 = self._make_stage(C2, C3, depths[2], stride=2, k=k)
        
        # Stage 4: 8× channels, anti-aliased downsample
        self.layer4 = self._make_stage(C3, C4, depths[3], stride=2, k=k)
        
        # Rhythm context head (optional)
        rhythm_dim = 128
        if self.use_rhythm_head:
            self.rhythm_head = RhythmContextHead(hidden_dim=rhythm_dim)
            print(f"[Model] ✓ Rhythm Context Head enabled (HRV + demographics → {rhythm_dim}D embedding)")
        else:
            self.rhythm_head = None
            print(f"[Model] Rhythm Context Head disabled")
        
        # Global pooling and classifier
        self.pool = nn.AdaptiveAvgPool1d(1)
        # Alias for evaluator hook compatibility
        self.avgpool = self.pool
        self.dropout = nn.Dropout(p=p_drop)
        
        # Classifier input dimension: C4 + rhythm_dim if rhythm head is enabled
        classifier_in_dim = C4 + rhythm_dim if self.use_rhythm_head else C4
        self.fc = CosineClassifier(classifier_in_dim, num_labels)

    def _make_stage(self, in_ch, out_ch, num_blocks, stride, k):
        """Build a simple ResNet stage with optional anti-aliased downsampling."""
        layers = []
        
        # Anti-aliased downsample before the stage if needed
        if stride == 2:
            layers.append(BlurPool1D(in_ch, stride=2))
            first_block_stride = 1
        else:
            first_block_stride = stride
        
        # First block handles channel change (no stride if blurpool used)
        layers.append(SimpleResBlock(in_ch, out_ch, stride=first_block_stride, k=k, use_se=True))
        
        # Remaining blocks
        for _ in range(1, num_blocks):
            layers.append(SimpleResBlock(out_ch, out_ch, stride=1, k=k, use_se=True))
        
        return nn.Sequential(*layers)

    def forward(self, x, hrv_features=None, age=None, sex=None):
        """
        Forward pass
        
        Args:
            x: Input tensor [B, 12, T] (batch, leads, time)
            hrv_features: Optional [B, 12] HRV features
            age: Optional [B] or [B, 1] normalized age
            sex: Optional [B] or [B, 1] encoded sex
        
        Returns:
            logits: [B, num_labels] (raw class logits)
        """
        # Per-lead stem processing
        per_lead = [stem(x[:, i:i+1, :]) for i, stem in enumerate(self.stems)]
        x = torch.cat(per_lead, dim=1)
        
        # Lead mixer before fusion (if enabled)
        x = self.lead_mixer(x)
        # Simple cross-lead fusion
        x = self.fuse(x)
        
        # ResNet stages (simple sequential processing)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        # Global pooling
        x = self.pool(x).squeeze(-1)  # [B, C4]
        
        # Late fusion with rhythm context head (if enabled)
        if self.use_rhythm_head and self.rhythm_head is not None and hrv_features is not None:
            rhythm_out = self.rhythm_head(hrv_features, age, sex)  # [B, 128]
            x = torch.cat([x, rhythm_out], dim=1)  # [B, C4+128]
        
        x = self.dropout(x)
        logits = self.fc(x)           # [B, num_labels]
        
        return logits

    def forward_features(self, x, hrv_features=None, age=None, sex=None):
        """Compute pooled feature representation before dropout/classifier.
        Returns tensor of shape [B, C4] or [B, C4+128] if rhythm head is enabled.
        """
        # Per-lead stem processing
        per_lead = [stem(x[:, i:i+1, :]) for i, stem in enumerate(self.stems)]
        x = torch.cat(per_lead, dim=1)
        # Lead mixer before fusion (if enabled)
        x = self.lead_mixer(x)
        # Simple cross-lead fusion
        x = self.fuse(x)
        # ResNet stages
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        # Global pooling
        x = self.pool(x).squeeze(-1)
        
        # Late fusion with rhythm context head (if enabled)
        if self.use_rhythm_head and self.rhythm_head is not None and hrv_features is not None:
            rhythm_out = self.rhythm_head(hrv_features, age, sex)
            x = torch.cat([x, rhythm_out], dim=1)
        
        return x

    @torch.inference_mode()
    def predict_proba(self, x, hrv_features=None, age=None, sex=None):
        """Predict class probabilities"""
        return torch.softmax(self.forward(x, hrv_features, age, sex), dim=1)

    @torch.inference_mode()
    def predict(self, x, hrv_features=None, age=None, sex=None):
        """Predict class labels"""
        return torch.argmax(self.predict_proba(x, hrv_features, age, sex), dim=1)


class CosineClassifier(nn.Module):
    """Cosine similarity classifier with learnable temperature (scale).
    Produces logits = s * cos(theta(w, x)), with L2-normalized weights and features.
    """
    def __init__(self, in_features: int, num_classes: int, init_s: float = 10.0):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(num_classes, in_features))
        self.scale = nn.Parameter(torch.tensor(float(init_s)))

    def forward(self, x):
        x_norm = F.normalize(x, p=2, dim=1)
        w_norm = F.normalize(self.weight, p=2, dim=1)
        logits = self.scale * F.linear(x_norm, w_norm)
        return logits
