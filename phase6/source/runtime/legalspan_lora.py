"""LoRA adaptation of the user's original Phase 1 LegalSpan prompt-copy model.

The archived backbone and pointer mixture remain the forward implementation.
Only LoRA factors and the existing scalar-output route/copy gates are trainable.
The source boundary excludes every target token, including target BOS. LegalSpan
requires one variable-length example per forward; the public batch API loops
over rows and weights their losses by the number of supervised summary tokens.
"""
from __future__ import annotations

import base64
from bisect import bisect_right
import hashlib
import importlib
import json
import math
from pathlib import Path
import sys
import zlib

import torch
from torch import nn
from torch.nn import functional as F


TOKENIZER_POLICY = 'maximal whitespace / Unicode L,M / Nd / remaining runs; UTF8; no normalization'
# Unicode13 L*/M* and Nd ranges, from the official UnicodeData.txt. Embedding
# these categories preserves the archived segmentation under newer Python UCDs.
# Official file SHA256: bdbffbbfc8ad4d3a6d01b5891510458f3d36f7170422af4ea2bed3211a73e8bb
_UNICODE13_DATA = 'c-lo;Nwy>*2)s}(oIpIwe|`UJbOaLCqDVqbaX<vJ{{B{cw!iW3`*ZtF`#QL@zdlB?eWPgK;vW6WU-JItkNg?m=qvfr{Tu6+5A?m8CBJ+7_PFov>EB+r&&@XYC2weBRqpXc+&%+xBKGAm#5l8f`}oTtxAF0J`32sKkB`6PF@`hF<tN6{`t1k|dg)fE8_*nAAQ-$cy{0-n(J8c1yn5BGf%lf`)`o;67fqri#P{|VI^QLU0;9EBPB17oZH5^bb0Mrz_gNBrqS#l@y#xaZdzFN}(95<O=QbjbLb=1+g!o?A?PW(IwLB1&Bo%<G8d{g&w2$U~xN`^Fl)ap@zR~6#%V-PQLQ=@SB!vQ~ws-VSy_%=cLNDD)%?6~8F{03HKsDPjFuyT&^)fKEv09<ksGt4L4-8a)&62(*(JBdRB`OKau&`^_sJW5Q?j|)KUy?$U5H;$>vtg}T8ioO~gdoAX>af#EE*g+v!-Uwfb{Y)p)?HmNiPRFCoTN~ENiK5C!F&o~5o}BPPNessjKei*$GHNjUK|4k=|Yoe2{}>igAcYG>SeH(Q4*-OBUSbyBhq_RVv<4}i>W0PAjw6}KKIpmFp1Pl@V%p6sEoT6+AWa@VRg53X;YVIq6!63?aJoJw$_?<3))rkc&f&BNpMPVwRE<0Nx<m<(R+vT&bf1l1}1vF7u3&?P6X^HNc9q2!O#i@S1`2i!F>;}RpKE9*+P_{(-E6>p_V~;Aqj^PbQ%v959FaBs*#<}I*s~t){T3s3|ibFvpWv#637-}op$qZ)W=I-4)^wG1!-Ewk+vT0^wDWNJdJ0oC3W}cY|;{e)Vdt3^=zdDU>vB<ADBhtLIE_?a<Fc!LRk4E7cKT;*P_gZLCqFw8LX@IoQ=|H)Dm^d>$NVk&k0nQBA7w6gq%n%p>QD8XLt7L>Jut9d)+x;C?T$$m;=Wd<nA9B1JYP{d(WY{56*pPEQ4c#VJd`65@Z*m^ig_^8<S|OPf8zKEraqx5)SLm5^|!NAJ+Msv)?lfW)f9P^Z=xhO^$48l_pndo{}&RTCTESU<Ogutxi<hicTP5)oni)PCv*!jdK~1sU=r_S&ehysL!=)?lk|?%A!nK_Gyw}JyA(87(|d7i%}j)A*_;Ve#ZG(-ISAdewy}yMtz#!Gd5|)K~%@JcD?7YQq!d&4x=!AmriW1?$+vVC_s`zcyE{I0S52MpP2L~l3avA&0Q8{(u0`0&L5cZB`L&SLhs>joY88h1J@!*NeC=Ll8ZR%C|fXtXDmYR!9Yz}sY!ou$|(f1za)w1AI?)Z7Z|*E>VBVxdM%4GX;J1mYKhLFLAND5iTILS1ciet-5D*OMv$vLa4t|c>h4%PP#{%@%O9vbPB3hFgfRRhiC~JpB!w1GIoMz@7O%AiGl)!xb#=zT_?M&*>U0}|L7%(M5f~?`QEyzlYlrHBb)mY$x>G9=IBZD@4I&eQyeM4)19_oQ2S*(mH7}o~8v)EFDtQO;;7Z-yG@4^LPq)r!Rza$+!*H8I3@{U@dNE$nw#yP3=V0<;jn|@SWX7@BsTmoy#}rD~!6~*Q3CCDh=g)ZO*r~M!6CYfu+Y-zmnuT!Kyp`pHV>Q1dg@Q<Rn4fc68L2Ts@m^NgsdX-`b7~#jth<6vbL_I8@VC%%*8Vxoqpz|(V9;`{l1|6hW$PR&0<=p49bY$%%s4VDs{ua^@5vH59ZL7oGp7sb^bWHZH`YbTqqe$&Kd08YYzUb7B`LIkDz<PcoTt;22(~;)wxo09Q70UpAkZnV%c)DdmF>YvI6!p=;5Z--Z6OqlV$ZIY*v;KdK`pL6n8+({56FX^tUeJk5LGW33y+Gl!4Mt22UXd#qJj;d0LX;!3dkw6*|P2&xTYBA>V;rizU^g&!9e;c4cT<6wtOith^<?lO)x|i2?sS2XIBUTNRmSP3#}5E^Ch_m?ohPye!;9Sdg%lhNa*D~f$=ZNMNmJGA;kCUR0SO%kdoJpEo<e2IuGwy?`=v5y$6c|a@Qr`n|(<x!cHf-2yTAN(vaYJh^4Xc?oU$4zK|yX2J%*U)WMKH6!!-s{ZMQV?ZX_F5*dsQlN4ev8s}nWV=1+<nA?y)0R}CtZmys!5jKRlp7JMxksG}KO4#(Y1ATU3Di>&ZkKpKScLu2!FWRDRWn;lHdc&eWY*}CW2+lKcQkvCoN&@bvGAWsz_~j^&me|BfDD}L%K>gu-ziusZs6qZo<KtBY`sTB9KD%o^JIA|+Kg|C2RDT)Q!7qOOt?W<PkDj)p%Yq;Y#{C%bRQs5;mZ0`9Q)RPaRdN@HFmHGy7K9gDf{uqR@qQR93%5Y+(-+KGUpVWPg8|3>-};SQAH=H~U*6mqUEX|E?*CB~2hQotoQfOi&Co5;Sp~-{9tLg~zD|tQEeNi&-iGRF{^{0ic{%gq^ZXpX@3nvA_`a7elY^^;JFC2bjKfLQKX(c=<SC^?@*>uiAuC*`^*pJSj+Oig1R0yoRIN8Sx;IG+Z{lA2503IPSh|=o66m<m9=OT&!(Kg3#-)Xw&fD<$Fx|IN{`bltX+f#fp8Es(#>evss`(+TVg0h_|84fKEF-zBFW;CP)_+=)d`tJyy5uv@NUHvM-a#u*2`;k!Xgtgbk(L)bJ=eqK4QJG`UPJ+Iay|Ca_;}tx-}-FN_OAKt&HYilK+cFOFHLSHmb`Q<XFPe4<9VLuOjmQp>pwi(hsKlc=28xT+(<khc4F-}{C+rkrYIU5>OpJ%;OJ`FsPpMLUu3f6E3I_naIDG;Nyp!*_QKF_QnhE;3|hV<Il5)C=j29WfB4qN^XbyXtMf%}CSDJFI(^UU!<`2IZb*Kl8>pCx9OrZD=7SSRm(E(3FG!~xbUw`a;X08&$nykRH#|8yk7GaY@vxq^_5C8j@E^3+7aZ&*=+^t#VBqyI6SJg}ziz#}KI0`zY7cY#%Flw!@!3*)+FQ^%59G4`JbW&G?GM{~=?^*9yVlbfe*23);1P2^uKc%O=N&22>fcAY%)`_^-Fa7z#hqRw=+yg@<Mmk?8E|ymLANlSXDVFbboza{JoM#RuZYAg`<LfIpPl_}>96y?B`yEkX|F-G-@bD`@!se81T~E}Ql2u0@C!J<J?s2~JH$)tnNBWf`4ez*GZ-*MG%ze|Z)ZHrdS#Z$9b)MpxU8?(m)s_H^+*2%lW6LX{s!n0ru_lcdTwTCNzaQ__<m@8cjWw(k??fI{rTJAW`;AGUogEuTKxsSRDGo5VZPtBe!&Qs^tB(^em%^<D)beF-1#mo%=Lr$tn~r6h)wba!?aLtf*j>d%^z9l47r->`E3kT>mM0b!dF)N3?<?0sy;)+P)5Te^(o$Yi5u>P;>1Y(3HdYNxIb|eCm)7gpwf>+%S<{R_QZ+`8DPQt)cHHi5?)%E>v44d`r!vTP@Tua`w?_J%=2}g{v3Lup!Mwr9M}JZ?Da`~Im+BKT+N_0G#}nqh+U<>;8=f5_4MROHDBr%!rWRPa{gUfxF@dogd79mR_jA<CANpzzL$L_cd`1TK3-aQ@%>)<Pi_*IK4V?-{qf$XWh2%5gCTBG;j9mP7~yeKc8nbRpIzhU{RjH`qx}P|>;WTv)~~ld)LhW_K6jqfALjZjeY|e;<Vn>(<3-Z`@J#$N3zqNuYY$6F`sD4Dw_p35|9bOZIp6hi6O0#0wVt?u`N{pQKmBu3jh{&u7}=@*@>6`{7p=P<`3W9W>#=ja;$d3uOAC9}mp_ZRjH#2ZFSz8;fA7j|!2MzRq)*jn0;P|Nxyku=Y2lUi%A52|mGo(9Pycx5_xM|%Nsz9no!lfYe+3-%tII#$8Gqt7EB^v<WR73<gdF=jUGqD*X+f9dpLMZ4%<-O=A7TcERQ+cn2A{`|FzgZO>+djW^v>(Z^Eu>yg5&&0fAx9(@KwAE!?J_=!#r=GOPFadgfGG)>lL4X<NA!L`3(9KmlmczjHmw@Z-VN)40`^cnlBtX{_R6wK6l2WpgMnpmJjr0$!OU@x5iJ+0xkQ$^Xnk#4`(9HH9qEGcYeyd88UvxyR<Ov!TYX9ZW7OjSJqdS9-L)r&R@M}dw=~q>vbQKTZ!#qrtkF4kGVbe--r5NDQ~&z`LX8G%J=cSEZ6#H)*NTF@)dhCzQO-9${)pb%#_*tx`cbs`Ii@c+ddgjgSzUuGobsPFI<$7f03zph1caE?rM(~#O0rXJ6W$k;Y`K@pq1kVhs?lv@isV|N6^||Ubm1pm2WwdiFDj<8vo+XfR?>GOjQ8)@+VJzVj<lRAL6o?;5a_e>VMIn?l@jbsDPy{!T*2iNcMLBNQ1{xlBuOXiA!xO&D2ijm0;&yoBWc5hxJ#IxiuIod&#z&joUB%l^@Iah?kJtDQoIOO0tn9eZLawZ+~Gf+`Y~Sn-BYyoGh{RN_figc+$aEiPCvc7>*eaCGgIqZu0T^x#NKid(u<?m`r+Y@`UoSU&+p%&rkZf($D_A#eKvher+P9pYz6(x>gUBFz5E1oMbIa^min%)HpuYujIZJzY<!R@tmJznLTTLDBadFGsK7gN+L_<dy7eIJydertY67WSu2#NYncZgd0|&c{s#<?mkew-8`$)xon%LroHrgj*^iI+mS@hg=}$Asru1Lhu=$m~f717rzJFuhcg*`IeP8K&zje;c4G!6?zo9{q*a82=rSbIiw(zj=N@!o}Q6)SV^{5j1IJn>5F89oikFLK=Ax)cK?#8<=l8wyYw)GOaE&tqj`Y**x?|wWZm+XB@=WAphkPm4l*_9<nsoVU6MLtf=J^xmrN2P=BTQ4C_c4UeExW|8tQMxG!OE#rvvP31>k15ee)^AC<V$A$!iOcy(_GF394%vN6v^$=;kb0<OetM|HvgFO{e*fd+yb>OodZ^^~Gk*QdD{KB_&D--!U^ChC>7kPI=}{%?%38mk*X?KduOvP_s>JSF_8Xzy5!$aGyZzX2Ja)%pzwy`|kNw8uw`cuPlkEERP|59Q|K^^jiP@o75_f&#H=elTiC>#@+nirNC;QQ7*JN{E{@WcN*^iI?E6G{-e6sxRg=>9qxl>p1kgn3bLCJo9>7XY~wS`64`v36GKZC+BH2uXi@)nNy{w@kO|1jDWSkt5C?TOn|vpD(r7?x9ediY;ecn`R}GW_}XA5rDIkp'
_CATEGORY_RANGES = json.loads(zlib.decompress(base64.b85decode(_UNICODE13_DATA)))
_CATEGORY_STARTS = [row[0] for row in _CATEGORY_RANGES]
_WHITESPACE = frozenset('\t\n\v\f\r\x1c\x1d\x1e\x1f \x85\xa0\u1680\u2000\u2001\u2002\u2003\u2004\u2005\u2006\u2007\u2008\u2009\u200a\u2028\u2029\u202f\u205f\u3000')
_CUES = frozenset(('section', 'sections', 'sec', 'article', 'articles', 'act', 'rule',
                   'rules', 'clause', 'clauses', 'subsection', 'scc', 'air', 'crpc', 'ipc', 'cpc'))


def _char_class(char):
    if char in _WHITESPACE:
        return 0
    cp = ord(char)
    index = bisect_right(_CATEGORY_STARTS, cp) - 1
    if index >= 0 and cp <= _CATEGORY_RANGES[index][1]:
        return _CATEGORY_RANGES[index][2]
    return 3


def _pieces(text):
    start, kind = 0, None
    for index, char in enumerate(text):
        current = _char_class(char)
        if kind is not None and current != kind:
            yield text[start:index]
            start = index
        kind = current
    if text:
        yield text[start:]


class TokenizerAdapter:
    """Exact archived manual byte-BPE merges with pinned Unicode13 boundaries."""
    pad_id, bos_id, eos_id, unk_id = 0, 1, 2, 3

    def __init__(self, path):
        from legalspan.bpe import OwnBPE
        record = json.loads(Path(path).read_text(encoding='utf-8'))
        if (record.get('format') != 'cs781-own-byte-bpe-v1'
                or record.get('policy') != TOKENIZER_POLICY
                or record.get('unicode_version') != '13.0.0'):
            raise ValueError('Expected the Phase1 manual byte-BPE tokenizer with Unicode13 segmentation')
        self.raw = OwnBPE(record['merges'])
        self.bytes = self.raw.bytes

    def encode(self, text):
        return [token for piece in _pieces(text)
                for token in self.raw._encode_piece(piece.encode('utf-8'))]

    def decode(self, ids):
        return self.raw.decode([int(token) for token in ids])

    def get_vocab_size(self):
        return len(self.bytes)

    def roles(self, ids):
        # Inspect only observed prompt bytes. Surrogateescape preserves token
        # offsets if the source budget ends partway through a UTF8 character.
        ids = [int(i) for i in ids]
        raw = b''.join(self.bytes[i] for i in ids)
        spans, offset = [], 0
        for piece in _pieces(raw.decode('utf-8', errors='surrogateescape')):
            end = offset + len(piece.encode('utf-8', errors='surrogateescape'))
            kind = _char_class(piece[0])
            role = (0 if any(0xdc80 <= ord(c) <= 0xdcff for c in piece)
                    else 1 if kind == 2 else 2 if piece.lower() in _CUES
                    else 3 if kind == 1 and piece[0].isupper()
                    else 4 if kind == 3 or '\n' in piece else 0)
            spans.append((offset, end, role)); offset = end
        roles, offset, cursor = [], 0, 0
        for token in ids:
            if token < 4:
                roles.append(0)
                continue
            end = offset + len(self.bytes[token])
            while cursor < len(spans) and offset >= spans[cursor][1]:
                cursor += 1
            if cursor >= len(spans) or end > spans[cursor][1]:
                raise ValueError('Prompt byte-BPE token crosses a Unicode13 piece boundary')
            roles.append(spans[cursor][2]); offset = end
        return roles


class LoRALinear(nn.Module):
    """Frozen original linear plus alpha/r * B(Ax); zero B preserves base output."""
    def __init__(self, base, rank=8, alpha=16):
        super().__init__()
        self.base = base
        self.rank, self.alpha = rank, alpha
        self.in_features, self.out_features = base.in_features, base.out_features
        self.lora_a = nn.Parameter(torch.empty(rank, base.in_features,
                                             device=base.weight.device, dtype=base.weight.dtype))
        self.lora_b = nn.Parameter(torch.zeros(base.out_features, rank,
                                             device=base.weight.device, dtype=base.weight.dtype))
        nn.init.kaiming_uniform_(self.lora_a, a=math.sqrt(5))

    def forward(self, x):
        return self.base(x) + F.linear(F.linear(x, self.lora_a), self.lora_b) * (self.alpha / self.rank)


def _adapt(model, rank=8, alpha=16):
    model.requires_grad_(False)
    adapted = []
    for name, module in list(model.named_modules()):
        if isinstance(module, nn.Linear) and name != 'lm_head' and not name.endswith('gate'):
            parent_name, _, leaf = name.rpartition('.')
            parent = model.get_submodule(parent_name) if parent_name else model
            setattr(parent, leaf, LoRALinear(module, rank, alpha))
            adapted.append(name)
    gates = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and (name.endswith('.route.gate') or name == 'copy_gate'):
            module.requires_grad_(True)
            gates.append(name)
    return adapted, gates


def _unique_file(root, name):
    matches = sorted(Path(root).rglob(name))
    if len(matches) != 1:
        raise FileNotFoundError(f'Expected exactly one {name} recursively under {root}; found {matches}')
    return matches[0]


def _import_source(input_dir):
    root = Path(input_dir)
    candidates = [root, root / 'legalspan_source', root / 'source']
    candidates.extend(path.parent.parent for path in root.rglob('legalspan/model.py'))
    for candidate in candidates:
        if (candidate / 'legalspan/model.py').is_file() and (candidate / 'cs781/models/a.py').is_file():
            candidate = candidate.resolve()
            # B1's separately archived source may already own these generic
            # package names in a shared test process. Bind both namespaces to
            # this checkpoint's verified source, never a previous ZIP import.
            incompatible = any(
                name in sys.modules
                and Path(getattr(sys.modules[name], '__file__', '')).resolve()
                    != candidate / name / '__init__.py'
                for name in ('cs781', 'legalspan'))
            if incompatible:
                for name in list(sys.modules):
                    if name in ('cs781', 'legalspan') or name.startswith(('cs781.', 'legalspan.')):
                        del sys.modules[name]
            sys.path[:] = [entry for entry in sys.path if entry != str(candidate)]
            sys.path.insert(0, str(candidate))
            importlib.invalidate_caches()
            break
    try:
        return importlib.import_module('legalspan.model')
    except ImportError as error:
        raise ImportError('Extract source/legalspan and source/cs781 into input_dir/legalspan_source, or add that source root to sys.path') from error


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def build(input_dir, device):
    original = _import_source(input_dir)
    checkpoint_path = _unique_file(input_dir, 'legalspan_phase1.pt')
    tokenizer_path = _unique_file(input_dir, 'legalspan_tokenizer.json')
    tokenizer = TokenizerAdapter(tokenizer_path)
    # Trusted user-owned Phase1 artifact. weights_only also accepts its pure
    # model/metadata dictionary and does not import arbitrary pickle globals.
    payload = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
    provenance = payload.get('metadata', {})
    config = provenance.get('model_config', payload.get('model_config'))
    if not config:
        raise ValueError('Phase1 checkpoint must contain metadata.model_config')
    cfg = original.SpanConfig(**config)
    if cfg.route_mode != 'full' or not all((cfg.use_copy, cfg.use_roles, cfg.use_continuity, cfg.use_fusion)):
        raise ValueError('Expected full LegalSpan prompt role/continuity/fusion/copy checkpoint')
    if tokenizer.get_vocab_size() != cfg.vocab_size:
        raise ValueError('Phase1 tokenizer vocabulary does not match checkpoint')
    tokenizer_hash = _sha256(tokenizer_path)
    if provenance.get('tokenizer_sha256', tokenizer_hash) != tokenizer_hash:
        raise ValueError('Phase1 checkpoint tokenizer SHA256 mismatch')
    model = original.LegalSpanGPT(cfg)
    model.load_state_dict(payload['model'], strict=True)
    del payload
    adapted, gates = _adapt(model)
    model._phase6_tokenizer = tokenizer
    model = model.to(device)
    metadata = {
        'architecture': 'Phase1-LegalSpan-full-LoRA-abstractive',
        'checkpoint_path': str(checkpoint_path), 'checkpoint_sha256': _sha256(checkpoint_path),
        'tokenizer_path': str(tokenizer_path), 'tokenizer_sha256': tokenizer_hash,
        'source_path': str(Path(original.__file__).resolve().parent.parent),
        'source_model_sha256': _sha256(original.__file__),
        'source_backbone_sha256': _sha256(importlib.import_module('cs781.models.a').__file__),
        'source_bpe_sha256': _sha256(importlib.import_module('legalspan.bpe').__file__),
        'adapter_sha256': _sha256(__file__),
        'model_config': config, 'lora_rank': 8, 'lora_alpha': 16,
        'lora_modules': adapted, 'trainable_original_gate_modules': gates,
        'total_parameters': sum(p.numel() for p in model.parameters()),
        'trainable_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad),
        'tokenizer_format': 'cs781-own-byte-bpe-v1', 'tokenizer_unicode_version': '13.0.0',
        'special_ids': {'pad': 0, 'bos': 1, 'eos': 2, 'unk': 3},
        'loss': 'token-weighted summary next-token mixture NLL; source and target BOS masked; EOS supervised',
        'prompt_scope': 'unpadded source only; summary BOS and tokens excluded from pointer memory',
        'generation': 'greedy cached decoding until EOS or max_new_tokens; no word-count contract',
    }
    return model, tokenizer, metadata


def _roles(model, ids):
    return torch.tensor([model._phase6_tokenizer.roles(ids.tolist())], device=ids.device, dtype=torch.long)


def _validate_batch(ids, mask, name):
    if ids.ndim != 2 or mask.shape != ids.shape:
        raise ValueError(f'{name} requires IDs and mask with matching [B,L] shapes')


def training_loss(model, source_ids, source_mask, target_ids, target_mask):
    _validate_batch(source_ids, source_mask, 'source')
    _validate_batch(target_ids, target_mask, 'target')
    if source_ids.shape[0] != target_ids.shape[0] or not source_ids.shape[0]:
        raise ValueError('Nonempty matching batch sizes required')
    losses, counts = [], []
    tok = model._phase6_tokenizer
    for src, sm, tgt, tm in zip(source_ids, source_mask, target_ids, target_mask):
        src, tgt = src[sm.bool()], tgt[tm.bool()]
        if not src.numel() or tgt.numel() < 2 or tgt[0].item() != tok.bos_id or tgt[-1].item() != tok.eos_id:
            raise ValueError('Nonempty source and targets containing BOS, summary, EOS required')
        inputs = torch.cat((src, tgt[:-1]))[None]
        labels = torch.full_like(inputs, -100)
        labels[:, src.numel():] = tgt[None, 1:]
        _, loss, _ = model(inputs, targets=labels, prompt_length=src.numel(),
                           prompt_roles=_roles(model, src), return_probs=False)
        count = tgt.numel() - 1
        losses.append(loss * count); counts.append(count)
    return torch.stack(losses).sum() / sum(counts)


@torch.no_grad()
def generate(model, source_ids, source_mask, max_new_tokens, eos_id):
    _validate_batch(source_ids, source_mask, 'source')
    if max_new_tokens < 0:
        raise ValueError('max_new_tokens must be nonnegative')
    was_training = model.training
    model.eval()
    outputs = []
    try:
        for src, mask in zip(source_ids, source_mask):
            src = src[mask.bool()]
            if not src.numel():
                raise ValueError('Generation requires nonempty source')
            if src.numel() + max_new_tokens > model.cfg.max_seq_len:
                raise ValueError('Source plus generation exceeds original LegalSpan context capacity')
            generated, cache = [], None
            current = torch.cat((src, src.new_tensor([model._phase6_tokenizer.bos_id])))[None]
            for _ in range(max_new_tokens):
                scores, _, cache = model(current, prompt_length=src.numel(),
                                          prompt_roles=_roles(model, src) if cache is None else None,
                                          cache=cache, use_cache=True)
                token = scores[0, -1].argmax().item()
                generated.append(token)
                if token == eos_id:
                    break
                current = src.new_tensor([[token]])
            outputs.append(generated)
    finally:
        model.train(was_training)
    return outputs
