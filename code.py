import argparse
import random
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import GroupKFold


FEATURES = [
    "sex",
    "handId",
    "strengthId",
    "spinId",
    "positionId",
    "strikeId",
    "scoreDiff_cat",
    "score_state",
    "is_deuce",
    "is_critical_point",
    "player_turn",
    "strikeNumber",
    "rally_phase",
    "serve_phase",
    "long_rally",
    "actionId", "pointId",
    "prev_actionId",
    "prev2_actionId",
    "prev_pointId",
    "action_transition"
]

PAD_TOKEN = 0

def add_features(df):
    diff = df["scoreSelf"] - df["scoreOther"]
    df["scoreDiff_cat"] = diff.clip(-5, 5) + 5

    df["score_state"] = 0
    df.loc[df["scoreSelf"] < df["scoreOther"], "score_state"] = 1
    df.loc[df["scoreSelf"] > df["scoreOther"], "score_state"] = 2
    df["is_deuce"] = ((df["scoreSelf"] >= 10) & (df["scoreOther"] >= 10)).astype(int)
    df["is_critical_point"] = ((df["scoreSelf"] >= 9) | (df["scoreOther"] >= 9)).astype(int)

    df["player_turn"] = (df["strikeNumber"] % 2).astype(int)
    df["serve_phase"] = (df["strikeNumber"] <= 3).astype(int)
    df["long_rally"] = (df["strikeNumber"] >= 8).astype(int)
    df["rally_phase"] = pd.cut(df["strikeNumber"], bins=[0, 3, 5, 100], labels=[0, 1, 2], right=True).astype(int)

    df["prev_actionId"] = df.groupby("rally_uid")["actionId"].shift(1).fillna(0).astype(int)
    df["prev_pointId"]  = df.groupby("rally_uid")["pointId"].shift(1).fillna(0).astype(int)
    df["prev2_actionId"] = df.groupby("rally_uid")["actionId"].shift(2).fillna(0).astype(int)

    df["action_transition"] = (
        df["prev_actionId"].astype(str)
        + "_"
        + df["actionId"].astype(str)
    ).astype("category").cat.codes + 1

    return df

class RallyDataset(Dataset):
    def __init__(self, X, yA, yP, yR, L):
        self.X = torch.tensor(X, dtype=torch.long)
        self.yA = torch.tensor(yA, dtype=torch.long)
        self.yP = torch.tensor(yP, dtype=torch.long)
        self.yR = torch.tensor(yR, dtype=torch.float32)
        self.L  = torch.tensor(L,  dtype=torch.long)
    def __len__(self): return self.X.shape[0]
    def __getitem__(self, i): return self.X[i], self.yA[i], self.yP[i], self.yR[i], self.L[i]

import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=100):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.pe = pe.unsqueeze(0) # shape: (1, max_len, d_model)

    def forward(self, x):
        # x shape: (batch_size, seq_len, d_model)
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len, :].to(x.device)

class MultiTaskTransformer(nn.Module):
    def __init__(self, num_tokens_per_feature, n_act, n_pt, emb_dim=32, hidden=128, num_layers=2, nhead=4, dropout=0.15):
        super().__init__()

        self.embs = nn.ModuleList([nn.Embedding(n+1, emb_dim, padding_idx=PAD_TOKEN) for n in num_tokens_per_feature])
        self.d_model = len(num_tokens_per_feature) * emb_dim
        self.input_proj = nn.Linear(self.d_model, hidden)
        self.pos_encoder = PositionalEncoding(d_model=hidden)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden, nhead=nhead, dim_feedforward=hidden * 4,
            dropout=dropout, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.norm = nn.LayerNorm(hidden)

        self.attention_pool = nn.Linear(hidden, 1, bias=False)
        self.drop = nn.Dropout(dropout)

        self.act_head = nn.Linear(hidden, n_act)


        self.pt_main_head = nn.Linear(hidden, n_pt)
        self.pt_side_head = nn.Linear(hidden, 4)
        self.pt_depth_head = nn.Linear(hidden, 4)


        self.rly_head = nn.Linear(hidden * 2, 1)

    def generate_causal_mask(self, sz):
        mask = torch.triu(torch.ones(sz, sz), diagonal=1).bool()
        return mask

    def forward(self, X, lengths):
        device = X.device
        batch_size, seq_len, _ = X.size()

        es = [emb(X[:,:,i]) for i, emb in enumerate(self.embs)]
        x = torch.cat(es, dim=-1)
        x = self.input_proj(x)
        x = self.pos_encoder(x)

        padding_mask = (X.sum(dim=-1) == 0)
        causal_mask = self.generate_causal_mask(seq_len).to(device)

        o = self.transformer(
            x, mask=causal_mask, src_key_padding_mask=padding_mask
        )



        o = self.norm(o)
        o = self.drop(o)


        valid_mask = (X[:, :, 0] != PAD_TOKEN).float().unsqueeze(-1)
        attn_weights = self.attention_pool(o)
        attn_weights = attn_weights.masked_fill(valid_mask == 0, -1e9)
        attn_weights = torch.softmax(attn_weights, dim=1)
        attn_hidden = (o * attn_weights).sum(dim=1)


        batch_idx = torch.arange(batch_size, device=device)
        last_hidden = o[batch_idx, lengths - 1]


        rally_feat = torch.cat([attn_hidden, last_hidden], dim=-1)


        return self.act_head(o), self.pt_main_head(o), self.pt_side_head(o), self.pt_depth_head(o), self.rly_head(rally_feat).squeeze(1)

def pad2d(a, m, pad_val=PAD_TOKEN):
    out = np.full((m, a.shape[1]), pad_val, dtype=np.int64); out[:len(a)] = a; return out
def pad1d(a, m, ignore_index=-1):
    out = np.full((m,), ignore_index, dtype=np.int64); out[:len(a)] = a; return out

import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, weight=None, gamma=2.0, ignore_index=-1):
        super(FocalLoss, self).__init__()
        self.weight = weight
        self.gamma = gamma
        self.ignore_index = ignore_index

    def forward(self, inputs, targets):


        ce_loss = F.cross_entropy(inputs, targets, reduction='none', ignore_index=self.ignore_index)


        pt = torch.exp(-ce_loss)


        focal_loss = ((1 - pt) ** self.gamma) * ce_loss


        if self.weight is not None:

            valid_mask = targets != self.ignore_index
            valid_targets = targets[valid_mask]


            weights = self.weight[valid_targets]
            focal_loss[valid_mask] = focal_loss[valid_mask] * weights


        return focal_loss[targets != self.ignore_index].mean()

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main(args):
    seed_everything(args.seed)
    train = pd.read_csv(args.train).sort_values(["rally_uid","strikeNumber"])
    test  = pd.read_csv(args.test).sort_values(["rally_uid","strikeNumber"])
    sub   = pd.read_csv(args.sample)

    train = add_features(train)
    test  = add_features(test)
    train["strikeNumber"] = train["strikeNumber"].clip(0, 40)
    test["strikeNumber"]  = test["strikeNumber"].clip(0, 40)

    cats = {c: pd.Categorical(train[c]).categories for c in FEATURES}
    def encode_frame(df):
        outs = []
        for col in FEATURES:
            codes = pd.Categorical(df[col], categories=cats[col]).codes + 1
            outs.append(np.asarray(codes, dtype=np.int64))
        return np.stack(outs, axis=1)

    X_list, yA_list, yP_list, yR_list, L_list, M_list = [], [], [], [], [], []
    for rid, g in train.groupby("rally_uid"):
        max_len = len(g)
        if max_len < 2: continue

        full_X = encode_frame(g)
        full_yA = g["actionId"].values.astype(np.int64)
        full_yP = g["pointId"].values.astype(np.int64)
        rally_winner = int(g["serverGetPoint"].iloc[0])
        match_id = int(g["match"].iloc[0])


        X_list.append(full_X[:-1])
        yA_list.append(full_yA[1:])
        yP_list.append(full_yP[1:])
        yR_list.append(rally_winner)
        L_list.append(max_len - 1)
        M_list.append(match_id)


        if max_len > 4:
            cut_idx = random.randint(2, max_len - 2)
            X_list.append(full_X[:cut_idx])
            yA_list.append(full_yA[1:cut_idx+1])
            yP_list.append(full_yP[1:cut_idx+1])
            yR_list.append(rally_winner)
            L_list.append(cut_idx)
            M_list.append(match_id)

    MAXLEN = max(L_list)
    X_all  = np.stack([pad2d(s, MAXLEN) for s in X_list])
    yA_all = np.stack([pad1d(s, MAXLEN) for s in yA_list])
    yP_all = np.stack([pad1d(s, MAXLEN) for s in yP_list])
    yR_all = np.array(yR_list, dtype=np.float32)
    L_all  = np.array(L_list, dtype=np.int64)
    M_all  = np.array(M_list, dtype=np.int64) # Match Array

    act_classes = np.sort(train["actionId"].unique()); n_act = len(act_classes); act_id2idx = {v:i for i,v in enumerate(act_classes)}
    pt_classes  = np.sort(train["pointId"].unique());  n_pt  = len(pt_classes);  pt_id2idx  = {v:i for i,v in enumerate(pt_classes)}
    yA_all = np.vectorize(act_id2idx.get)(yA_all, -1)
    yP_all = np.vectorize(pt_id2idx.get)(yP_all, -1)
    num_tokens_per_feature = [len(cats[c]) + 1 for c in FEATURES]



    gkf = GroupKFold(n_splits=5)


    fold_model_paths = []


    for fold, (tr_idx, va_idx) in enumerate(gkf.split(X_all, yR_all, groups=M_all)):
        print(f"\n{'='*20} Training Fold {fold+1}/5 {'='*20}")

        X_tr, X_va = X_all[tr_idx], X_all[va_idx]
        yA_tr, yA_va = yA_all[tr_idx], yA_all[va_idx]
        yP_tr, yP_va = yP_all[tr_idx], yP_all[va_idx]
        yR_tr, yR_va = yR_all[tr_idx], yR_all[va_idx]
        L_tr,  L_va  = L_all[tr_idx],  L_all[va_idx]


        act_counts = np.bincount(yA_tr[yA_tr!=-1].ravel(), minlength=n_act) + 1
        pt_counts  = np.bincount(yP_tr[yP_tr!=-1].ravel(), minlength=n_pt) + 1
        act_w = torch.tensor(1.0 / (act_counts ** 0.5), dtype=torch.float32)
        act_w = (act_w * (n_act / act_w.sum()))
        pt_w  = torch.tensor(1.0 / (pt_counts ** 0.5),  dtype=torch.float32)
        pt_w  = (pt_w  * (n_pt / pt_w.sum()))

        train_ds = RallyDataset(X_tr, yA_tr, yP_tr, yR_tr, L_tr)
        val_ds   = RallyDataset(X_va, yA_va, yP_va, yR_va, L_va)
        train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True)
        val_loader   = DataLoader(val_ds,   batch_size=max(args.batch*2,128), shuffle=False)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


        model = MultiTaskTransformer(
          num_tokens_per_feature, n_act, n_pt,
          emb_dim=args.emb, hidden=args.hidden,
          num_layers=args.layers, nhead=4, dropout=args.drop
        ).to(device)

        opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

        ce_action = nn.CrossEntropyLoss(ignore_index=-1, weight=act_w.to(device), label_smoothing=0.02)
        ce_point_main = FocalLoss(weight=pt_w.to(device), gamma=1.5, ignore_index=-1)
        ce_point_side = FocalLoss(gamma=1.5, ignore_index=-1)
        ce_point_depth = FocalLoss(gamma=1.5, ignore_index=-1)
        bce_rally = nn.BCEWithLogitsLoss()

        best_final_score = -1.0
        patience = 7
        no_improve_count = 0
        best_model_path = f"best_model_fold_{fold}.pt"
        fold_model_paths.append(best_model_path)

        for ep in range(1, args.epochs+1):
            model.train()
            run_loss = 0.0

            for Xb,yAb,yPb,yRb,Lb in train_loader:
              Xb,yAb,yPb,yRb,Lb = Xb.to(device),yAb.to(device),yPb.to(device),yRb.to(device),Lb.to(device)
              opt.zero_grad();
              la, lp_main, lp_side, lp_depth, lr = model(Xb, Lb)


              valid_mask = (yPb != -1)
              yPb_safe = yPb.clone()
              yPb_safe[~valid_mask] = 0
              side_map = torch.tensor([0, 1, 2, 3, 1, 2, 3, 1, 2, 3], dtype=torch.long, device=device)
              depth_map = torch.tensor([0, 1, 1, 1, 2, 2, 2, 3, 3, 3], dtype=torch.long, device=device)
              yPb_side = side_map[yPb_safe]
              yPb_depth = depth_map[yPb_safe]
              yPb_side[~valid_mask] = -1
              yPb_depth[~valid_mask] = -1


              lossA = ce_action(la.view(-1, la.size(-1)), yAb.view(-1))


              lossP_main = ce_point_main(lp_main.view(-1, lp_main.size(-1)), yPb.view(-1))
              lossP_side = ce_point_side(lp_side.view(-1, 4), yPb_side.view(-1))
              lossP_depth = ce_point_depth(lp_depth.view(-1, 4), yPb_depth.view(-1))
              lossP = 0.6 * lossP_main + 0.2 * lossP_side + 0.2 * lossP_depth

              lossR = bce_rally(lr, yRb)


              loss = 0.45 * lossA + 0.45 * lossP + 0.10 * lossR
              raw_loss = loss

              loss.backward()
              torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
              opt.step()


              run_loss += raw_loss.item() * Xb.size(0)

            scheduler.step()

            model.eval(); val_loss=0.0
            allA,allAp,allP,allPp,allR,allRp=[],[],[],[],[],[]
            with torch.no_grad():
              for Xb,yAb,yPb,yRb,Lb in val_loader:
                Xb,yAb,yPb,yRb,Lb = Xb.to(device),yAb.to(device),yPb.to(device),yRb.to(device),Lb.to(device)

                la, lp_main, lp_side, lp_depth, lr = model(Xb, Lb)

                valid_mask = (yPb != -1)
                yPb_safe = yPb.clone()
                yPb_safe[~valid_mask] = 0
                side_map = torch.tensor([0, 1, 2, 3, 1, 2, 3, 1, 2, 3], dtype=torch.long, device=device)
                depth_map = torch.tensor([0, 1, 1, 1, 2, 2, 2, 3, 3, 3], dtype=torch.long, device=device)
                yPb_side = side_map[yPb_safe]
                yPb_depth = depth_map[yPb_safe]
                yPb_side[~valid_mask] = -1
                yPb_depth[~valid_mask] = -1

                lossA = ce_action(la.view(-1, la.size(-1)), yAb.view(-1))
                lossP_main = ce_point_main(lp_main.view(-1, lp_main.size(-1)), yPb.view(-1))
                lossP_side = ce_point_side(lp_side.view(-1, 4), yPb_side.view(-1))
                lossP_depth = ce_point_depth(lp_depth.view(-1, 4), yPb_depth.view(-1))
                lossP = 0.6 * lossP_main + 0.2 * lossP_side + 0.2 * lossP_depth
                lossR = bce_rally(lr, yRb)

                raw_loss = 0.45 * lossA + 0.45 * lossP + 0.10 * lossR
                val_loss += raw_loss.item() * Xb.size(0)

                allR+=yRb.detach().cpu().tolist(); allRp+=torch.sigmoid(lr).detach().cpu().tolist()
                yA_flat=yAb.view(-1).detach().cpu().numpy(); yP_flat=yPb.view(-1).detach().cpu().numpy()

                a_pred = la.argmax(-1).view(-1).detach().cpu().numpy()
                p_pred = lp_main.argmax(-1).view(-1).detach().cpu().numpy()

                mA=(yA_flat!=-1); mP=(yP_flat!=-1)
                allA+=yA_flat[mA].tolist(); allAp+=a_pred[mA].tolist()
                allP+=yP_flat[mP].tolist(); allPp+=p_pred[mP].tolist()

              tr_loss = run_loss/len(train_loader.dataset); va_loss=val_loss/len(val_loader.dataset)
              try:
                f1A=f1_score(allA,allAp,average="macro") if len(allA) else 0.0
                f1P=f1_score(allP,allPp,average="macro") if len(allP) else 0.0
                auc=roc_auc_score(allR,allRp) if len(set(allR))>1 else 0.5
              except Exception:
                f1A,f1P,auc=0.0,0.0,0.5
              final=0.4*f1A+0.4*f1P+0.2*auc
              print(f"[Epoch {ep}/{args.epochs}] train_loss={tr_loss:.4f} val_loss={va_loss:.4f} F1_action={f1A:.4f} F1_point={f1P:.4f} AUC={auc:.4f} Final~{final:.4f}")

              if final > best_final_score:
                best_final_score = final
                no_improve_count = 0
                torch.save(model.state_dict(), best_model_path)
              else:
                no_improve_count += 1

              if no_improve_count >= patience:
                print(f" Early stopping triggered at epoch {ep} for Fold {fold+1}!")
                break
    models = []
    for path in fold_model_paths:
        m = MultiTaskTransformer(
            num_tokens_per_feature, n_act, n_pt,
            emb_dim=args.emb, hidden=args.hidden,
            num_layers=args.layers, nhead=4, dropout=args.drop
        ).to(device)
        m.load_state_dict(torch.load(path))
        m.eval()
        models.append(m)

    def pad2d_cap(a, m, pad_val=PAD_TOKEN):
        out = np.full((m, a.shape[1]), pad_val, dtype=np.int64)
        T = min(len(a), m); out[:T]=a[:T]; return out, T

    pred_rows=[]
    with torch.no_grad():
        for rid,g in test.groupby("rally_uid"):
            Xg = encode_frame(g); Xp,T = pad2d_cap(Xg, MAXLEN)
            X_t = torch.tensor(Xp[None,...], dtype=torch.long, device=device)
            L_t = torch.tensor([max(1,T)], dtype=torch.long, device=device)
            last_t = L_t.item() - 1

            la_preds = []
            lp_main_preds = []
            lr_probs = []

            for m in models:
                la, lp_main, lp_side, lp_depth, lr = m(X_t, L_t)

                la_preds.append(la[0, last_t])
                lp_main_preds.append(lp_main[0, last_t])
                lr_probs.append(torch.sigmoid(lr).item())

            avg_la = torch.stack(la_preds).mean(dim=0)
            avg_lp_main = torch.stack(lp_main_preds).mean(dim=0)
            avg_lr_prob = sum(lr_probs) / len(lr_probs)

            a_idx = int(torch.argmax(avg_la).item())
            p_idx = int(torch.argmax(avg_lp_main).item())

            action_pred = int(act_classes[a_idx])
            point_pred  = int(pt_classes[p_idx])

            pred_rows.append({
                "rally_uid": int(rid),
                "serverGetPoint": avg_lr_prob,
                "pointId": point_pred,
                "actionId": action_pred
            })

    pred_df = pd.DataFrame(pred_rows).sort_values("rally_uid")
    out = pred_df[["rally_uid", "actionId", "pointId", "serverGetPoint"]]
    out.to_csv(args.out, index=False)
    print(f"Saved submission to: {args.out}")
    print(out.head())

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default="train.csv")
    ap.add_argument("--test", default="test.csv")
    ap.add_argument("--sample", default="sample_submission.csv")
    ap.add_argument("--out", default="submission_lstm_baseline.csv")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--emb", type=int, default=24)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--layers", type=int, default=1)
    ap.add_argument("--drop", type=float, default=0.15)
    ap.add_argument("--lr", type=float, default=2e-4)

    ap.add_argument("--val_size", type=float, default=0.10)
    se = random.randint(0, 1000000)
    se = 42
    print(se)
    ap.add_argument("--seed", type=int, default=se)
    args = ap.parse_args(args=[])
    main(args)
