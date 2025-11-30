#!/usr/bin/env python3
"""
Script pour générer les tableaux LaTeX des performances avec et sans BatchNorm
Génère:
1. Un tableau avec deux sous-tableaux pour les F1-scores par classe
2. Un tableau avec deux sous-tableaux pour les meilleures performances
"""

import pandas as pd
import numpy as np
import sys
import argparse

def generate_f1_class_table(df, experiment_name="", label_suffix=""):
    """Génère un tableau LaTeX avec F1-scores par classe (avec 2 sous-tableaux)"""
    
    # Filtrer les expériences complétées
    df = df[df['status'] == 'completed']
    
    # Séparer les données avec et sans batchnorm
    with_bn = df[df['batchnorm'] == True]
    without_bn = df[df['batchnorm'] == False]
    
    # Identifier les colonnes de F1 par classe
    f1_class_columns = [col for col in df.columns if col.startswith('f1_class_')]
    classes = [(col, col.replace('f1_class_', '')) for col in f1_class_columns]
    
    print(r'\begin{table}[h]')
    print(r'\centering')
    print(f'\\caption{{Performance F1-Score par classe avec et sans BatchNorm{experiment_name}}}')
    print(r'\begin{subtable}{0.48\textwidth}')
    print(r'\centering')
    print(r'\caption{Performance F1-Score par classe AVEC BatchNorm}')
    print(f'\\label{{tab:f1_with_batchnorm{label_suffix}}}')
    print(r'\begin{tabular}{lc}')
    print(r'\toprule')
    print(r'Classe & F1-Score \\')
    print(r'\midrule')
    
    for metric, label in classes:
        mean_val = with_bn[metric].mean()
        std_val = with_bn[metric].std()
        print(f'{label} & ${mean_val:.4f} \\pm {std_val:.4f}$ \\\\')
    
    # Moyenne globale
    mean_global = with_bn['f1_score'].mean()
    std_global = with_bn['f1_score'].std()
    print(r'\midrule')
    print(f'Moyenne globale & ${mean_global:.4f} \\pm {std_global:.4f}$ \\\\')
    print(r'\bottomrule')
    print(r'\end{tabular}')
    print(r'\end{subtable}')
    print(r'\hfill')
    
    # Deuxième sous-tableau: SANS BatchNorm
    print(r'\begin{subtable}{0.48\textwidth}')
    print(r'\centering')
    print(r'\caption{Performance F1-Score par classe SANS BatchNorm}')
    print(f'\\label{{tab:f1_without_batchnorm{label_suffix}}}')
    print(r'\begin{tabular}{lc}')
    print(r'\toprule')
    print(r'Classe & F1-Score \\')
    print(r'\midrule')
    
    for metric, label in classes:
        mean_val = without_bn[metric].mean()
        std_val = without_bn[metric].std()
        print(f'{label} & ${mean_val:.4f} \\pm {std_val:.4f}$ \\\\')
    
    # Moyenne globale
    mean_global = without_bn['f1_score'].mean()
    std_global = without_bn['f1_score'].std()
    print(r'\midrule')
    print(f'Moyenne globale & ${mean_global:.4f} \\pm {std_global:.4f}$ \\\\')
    print(r'\bottomrule')
    print(r'\end{tabular}')
    print(r'\end{subtable}')
    print(r'\end{table}')


def generate_global_performance_table(df, label_suffix=""):
    """Génère un tableau comparant les performances moyennes globales avec et sans BatchNorm"""
    
    # Filtrer les expériences complétées
    df = df[df['status'] == 'completed']
    
    # Séparer les données avec et sans batchnorm
    with_bn = df[df['batchnorm'] == True]
    without_bn = df[df['batchnorm'] == False]
    
    # Metrics à analyser
    metrics = [
        ('test_accuracy', 'Test Accuracy (\\%)'),
        ('f1_score', 'F1 Score'),
        ('precision', 'Precision'),
        ('recall', 'Recall'),
        ('training_duration', 'Temps d\'entraînement (h)'),
        ('current_epoch', 'Nombre d\'epochs')
    ]
    
    print(r'\begin{table}[h]')
    print(r'\centering')
    print(r'\caption{Performance moyenne avec et sans BatchNorm sur l\'ensemble des configurations testées}')
    print(f'\\label{{tab:batchnorm_comparison{label_suffix}}}')
    print(r'\begin{tabular}{lccc}')
    print(r'\toprule')
    print(r'Métrique & Avec BatchNorm & Sans BatchNorm & Différence \\')
    print(r'\midrule')
    
    for metric, label in metrics:
        if metric not in df.columns:
            continue
            
        # Avec batchnorm
        mean_with = with_bn[metric].mean()
        std_with = with_bn[metric].std()
        
        # Sans batchnorm
        mean_without = without_bn[metric].mean()
        std_without = without_bn[metric].std()
        
        # Différence
        diff = mean_with - mean_without
        
        if metric == 'test_accuracy':
            # Pour accuracy, afficher en pourcentage
            print(f'{label} & ${mean_with:.2f} \\pm {std_with:.2f}$ & ${mean_without:.2f} \\pm {std_without:.2f}$ & ${diff:+.2f}$ \\\\')
        elif metric == 'training_duration':
            # Pour le temps, convertir en heures
            mean_with_h = mean_with / 3600
            std_with_h = std_with / 3600
            mean_without_h = mean_without / 3600
            std_without_h = std_without / 3600
            diff_h = diff / 3600
            print(f'{label} & ${mean_with_h:.2f} \\pm {std_with_h:.2f}$ & ${mean_without_h:.2f} \\pm {std_without_h:.2f}$ & ${diff_h:+.2f}$ \\\\')
        elif metric == 'current_epoch':
            # Pour les epochs, afficher en entier avec 1 décimale pour la variance
            print(f'{label} & ${mean_with:.1f} \\pm {std_with:.1f}$ & ${mean_without:.1f} \\pm {std_without:.1f}$ & ${diff:+.1f}$ \\\\')
        else:
            # Pour les autres métriques
            print(f'{label} & ${mean_with:.4f} \\pm {std_with:.4f}$ & ${mean_without:.4f} \\pm {std_without:.4f}$ & ${diff:+.4f}$ \\\\')
    
    print(r'\bottomrule')
    print(r'\end{tabular}')
    print(r'\end{table}')


def generate_best_performance_table(df, param_name="param_value", label_suffix=""):
    """Génère un tableau avec les meilleures performances (avec 2 sous-tableaux)"""
    
    # Filtrer les expériences complétées
    df = df[df['status'] == 'completed']
    
    # Séparer les données avec et sans batchnorm
    with_bn = df[df['batchnorm'] == True]
    try:
        without_bn = df[df['batchnorm'] == False]
    except KeyError:
        without_bn = pd.DataFrame()  # ou une autre gestion d'erreur appropriée
    
    # Trouver les meilleures performances (moyennes par paramètre)
    def get_best_config(data, param_col):
        if param_col not in data.columns:
            param_col = 'param_value'
        
        grouped = data.groupby(param_col).agg({
            'f1_score': ['mean', 'std'],
            'current_epoch': ['mean', 'std'],
            'training_duration': ['mean', 'std']
        })
        
        # Trouver la meilleure moyenne de F1
        best_idx = grouped[('f1_score', 'mean')].idxmax()
        best_row = grouped.loc[best_idx]
        
        return {
            'param_value': best_idx,
            'f1_mean': best_row[('f1_score', 'mean')],
            'f1_std': best_row[('f1_score', 'std')],
            'epoch_mean': best_row[('current_epoch', 'mean')],
            'epoch_std': best_row[('current_epoch', 'std')],
            'duration_mean': best_row[('training_duration', 'mean')],
            'duration_std': best_row[('training_duration', 'std')]
        }
    
    best_with = get_best_config(with_bn, param_name)
    try:
        best_without = get_best_config(without_bn, param_name)
    except :
        best_without = {}
    
    # Déterminer le nom du paramètre pour l'affichage
    param_display_name = param_name.replace('_', ' ').title()
    if param_name == 'param_value':
        param_display_name = 'Paramètre'
    elif param_name == 'hidden_size':
        param_display_name = 'Taille couche cachée'
    elif param_name == 'batch_size':
        param_display_name = 'Taille du batch'
    
    print()
    print()
    print(r'\begin{table}[h]')
    print(r'\centering')
    print(r'\caption{Meilleures performances avec et sans BatchNorm}')
    print(r'\begin{subtable}{0.48\textwidth}')
    print(r'\centering')
    print(r'\caption{Meilleure performance AVEC BatchNorm}')
    print(f'\\label{{tab:best_with_batchnorm{label_suffix}}}')
    print(r'\begin{tabular}{lc}')
    print(r'\toprule')
    print(r'Métrique & Valeur \\')
    print(r'\midrule')
    print(f'{param_display_name} & {best_with["param_value"]} \\\\')
    print(f'F1 Score & ${best_with["f1_mean"]:.4f} \\pm {best_with["f1_std"]:.4f}$ \\\\')
    print(f'Nb epochs & ${best_with["epoch_mean"]:.1f} \\pm {best_with["epoch_std"]:.1f}$ \\\\')
    
    # Convertir durée en heures
    duration_h_mean = best_with["duration_mean"] / 3600
    duration_h_std = best_with["duration_std"] / 3600
    print(f'Durée (h) & ${duration_h_mean:.2f} \\pm {duration_h_std:.2f}$ \\\\')
    print(r'\bottomrule')
    print(r'\end{tabular}')
    print(r'\end{subtable}')
    print(r'\hfill')
    
    try:
        # Deuxième sous-tableau: SANS BatchNorm
        print(r'\begin{subtable}{0.48\textwidth}')
        print(r'\centering')
        print(r'\caption{Meilleure performance SANS BatchNorm}')
        print(f'\\label{{tab:best_without_batchnorm{label_suffix}}}')
        print(r'\begin{tabular}{lc}')
        print(r'\toprule')
        print(r'Métrique & Valeur \\')
        print(r'\midrule')
        print(f'{param_display_name} & {best_without["param_value"]} \\\\')
        print(f'F1 Score & ${best_without["f1_mean"]:.4f} \\pm {best_without["f1_std"]:.4f}$ \\\\')
        print(f'Nb epochs & ${best_without["epoch_mean"]:.1f} \\pm {best_without["epoch_std"]:.1f}$ \\\\')
        
        # Convertir durée en heures
        duration_h_mean = best_without["duration_mean"] / 3600
        duration_h_std = best_without["duration_std"] / 3600
        print(f'Durée (h) & ${duration_h_mean:.2f} \\pm {duration_h_std:.2f}$ \\\\')
        print(r'\bottomrule')
        print(r'\end{tabular}')
        print(r'\end{subtable}')
        print(r'\end{table}')
    except:
        pass


def main():
    parser = argparse.ArgumentParser(description='Génère des tableaux LaTeX pour l\'analyse des performances')
    parser.add_argument('csv_file', help='Chemin vers le fichier CSV à analyser')
    parser.add_argument('--param-name', default='param_value', 
                        help='Nom de la colonne du paramètre varié (défaut: param_value)')
    parser.add_argument('--experiment-name', default='',
                        help='Nom de l\'expérience à ajouter dans le titre (ex: " (1 couche cachée [18])")')
    
    args = parser.parse_args()
    
    # Charger les données
    try:
        df = pd.read_csv(args.csv_file)
        print(f"% Données chargées depuis: {args.csv_file}", file=sys.stderr)
        print(f"% {len(df)} expériences trouvées", file=sys.stderr)
        print(f"% {len(df[df['status'] == 'completed'])} expériences complétées", file=sys.stderr)
        print("", file=sys.stderr)
    except Exception as e:
        print(f"Erreur lors du chargement du fichier: {e}", file=sys.stderr)
        sys.exit(1)
    
    # Créer un suffixe de label à partir de experiment_name
    # Remplacer les espaces et caractères spéciaux par des underscores
    label_suffix = ""
    if args.experiment_name:
        # Nettoyer le nom pour créer un label valide
        label_suffix = "_" + args.experiment_name.strip().replace(" ", "_").replace("(", "").replace(")", "").replace(",", "").replace("[", "").replace("]", "")
    
    # Générer le tableau des performances moyennes globales
    print("% ========================================")
    print("% Tableau 1: Performances moyennes globales")
    print("% ========================================")
    generate_global_performance_table(df, label_suffix)
    
    # Générer le tableau des F1-scores par classe
    print()
    print("% ========================================")
    print("% Tableau 2: F1-Scores par classe")
    print("% ========================================")
    generate_f1_class_table(df, args.experiment_name, label_suffix)
    
    # Générer le tableau des meilleures performances
    print()
    print("% ========================================")
    print("% Tableau 3: Meilleures performances")
    print("% ========================================")
    generate_best_performance_table(df, args.param_name, label_suffix)


if __name__ == "__main__":
    main()
