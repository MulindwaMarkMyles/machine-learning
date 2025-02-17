import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from scipy import signal
from scipy.stats import kurtosis, skew
import mplcursors

# Load and prepare the data
df = pd.read_csv("cleaned_dataset.csv")

# Rename columns to EEG channel names
eeg_channels = ['Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2', 
                'F7', 'F8', 'T7', 'T8', 'P7', 'P8', 'Fz', 'Cz', 'Pz']
df.columns = eeg_channels

# Define brain regions for grouping
frontal = ['Fp1', 'Fp2', 'F3', 'F4', 'F7', 'F8', 'Fz']
central = ['C3', 'C4', 'Cz']
parietal = ['P3', 'P4', 'P7', 'P8', 'Pz']
temporal = ['T7', 'T8']
occipital = ['O1', 'O2']

def create_eeg_visualizations(df):
    # 1. EEG Channel Distribution Overview
    plt.figure(figsize=(15, 10))
    df.boxplot()
    plt.xticks(rotation=45)
    plt.title('EEG Signal Distribution Across Channels')
    plt.ylabel('Amplitude (μV)')
    plt.show()

    # 2. Correlation Heatmap with EEG Channel Names
    plt.figure(figsize=(12, 10))
    sns.heatmap(df.corr(), annot=True, cmap='RdBu_r', 
                xticklabels=eeg_channels, yticklabels=eeg_channels)
    plt.title('EEG Channel Correlation Matrix')
    plt.show()

    # 3. Brain Region Average Signals
    region_means = {
        'Frontal': df[frontal].mean(axis=1),
        'Central': df[central].mean(axis=1),
        'Parietal': df[parietal].mean(axis=1),
        'Temporal': df[temporal].mean(axis=1),
        'Occipital': df[occipital].mean(axis=1)
    }
    
    plt.figure(figsize=(12, 6))
    for region, signal in region_means.items():
        plt.plot(signal[:1000], label=region, alpha=0.7)
    plt.title('Average Signal by Brain Region (First 1000 samples)')
    plt.legend()
    plt.xlabel('Time Points')
    plt.ylabel('Amplitude (μV)')
    plt.show()

    # 4. Channel Signal Distribution
    fig = plt.figure(figsize=(15, 10))
    for i, channel in enumerate(eeg_channels, 1):
        plt.subplot(4, 5, i)
        sns.kdeplot(data=df[channel], fill=True)
        plt.title(f'{channel} Distribution')
    plt.tight_layout()
    plt.show()

    # 5. Symmetry Analysis (Left vs Right)
    symmetry_pairs = [('Fp1', 'Fp2'), ('F3', 'F4'), ('C3', 'C4'), 
                     ('P3', 'P4'), ('O1', 'O2'), ('F7', 'F8'), ('T7', 'T8')]
    
    fig = plt.figure(figsize=(15, 10))
    for i, (left, right) in enumerate(symmetry_pairs, 1):
        plt.subplot(2, 4, i)
        plt.scatter(df[left], df[right], alpha=0.1)
        plt.xlabel(left)
        plt.ylabel(right)
        plt.title(f'{left} vs {right} Symmetry')
    plt.tight_layout()
    plt.show()

    # 6. Time Series Analysis
    plt.figure(figsize=(15, 8))
    for channel in ['Fz', 'Cz', 'Pz']:  # Midline channels
        plt.plot(df[channel][:500], label=channel, alpha=0.7)
    plt.title('Midline Channels Time Series (First 500 samples)')
    plt.legend()
    plt.xlabel('Time Points')
    plt.ylabel('Amplitude (μV)')
    plt.show()

    # 7. Brain Region Violin Plots
    plt.figure(figsize=(15, 8))
    region_data = []
    region_labels = []
    for region, channels in [('Frontal', frontal), ('Central', central), 
                           ('Parietal', parietal), ('Temporal', temporal), 
                           ('Occipital', occipital)]:
        for channel in channels:
            region_data.extend(df[channel].values)
            region_labels.extend([region] * len(df))
    
    sns.violinplot(x=region_labels, y=region_data)
    plt.title('Signal Distribution by Brain Region')
    plt.ylabel('Amplitude (μV)')
    plt.xticks(rotation=45)
    plt.show()

    # 8. Channel Variance Analysis
    variances = df.var().sort_values(ascending=False)
    plt.figure(figsize=(12, 6))
    variances.plot(kind='bar')
    plt.title('Channel Signal Variance')
    plt.xlabel('Channels')
    plt.ylabel('Variance')
    plt.xticks(rotation=45)
    plt.show()

    # 9. Power Spectral Density Plot
    plt.figure(figsize=(15, 8))
    for channel in ['Fz', 'Cz', 'Pz']:
        f, Pxx = signal.welch(df[channel], fs=256, nperseg=1024)
        plt.semilogy(f, Pxx)
        plt.xlabel('Frequency [Hz]')
        plt.ylabel('PSD [V**2/Hz]')
    plt.legend(['Fz', 'Cz', 'Pz'])
    plt.title('Power Spectral Density of Midline Channels')
    plt.show()

    # 10. Channel Kurtosis Analysis
    kurt_values = df.apply(kurtosis)
    plt.figure(figsize=(12, 6))
    sns.barplot(x=kurt_values.index, y=kurt_values.values)
    plt.title('Kurtosis Analysis by Channel')
    plt.xticks(rotation=45)
    plt.ylabel('Kurtosis Value')
    plt.show()

    # 11. Channel Skewness Analysis
    skew_values = df.apply(skew)
    plt.figure(figsize=(12, 6))
    sns.barplot(x=skew_values.index, y=skew_values.values)
    plt.title('Skewness Analysis by Channel')
    plt.xticks(rotation=45)
    plt.ylabel('Skewness Value')
    plt.show()

    # 12. Rolling Mean Analysis
    plt.figure(figsize=(15, 8))
    window_size = 100
    for channel in ['O1', 'O2']:  # Occipital channels
        rolling_mean = df[channel].rolling(window=window_size).mean()
        plt.plot(rolling_mean[:1000], label=f'{channel} Rolling Mean')
    plt.title(f'Rolling Mean Analysis (Window Size: {window_size})')
    plt.legend()
    plt.show()

    # 13. Channel Cross-Correlation Matrix
    cross_corr = np.zeros((len(eeg_channels), len(eeg_channels)))
    for i, ch1 in enumerate(eeg_channels):
        for j, ch2 in enumerate(eeg_channels):
            cross_corr[i,j] = np.correlate(df[ch1], df[ch2])[0]
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(cross_corr, xticklabels=eeg_channels, yticklabels=eeg_channels,
                cmap='viridis')
    plt.title('Channel Cross-Correlation Matrix')
    plt.show()

    # 14. Regional Signal Stability
    plt.figure(figsize=(15, 8))
    for region, channels in [('Frontal', frontal), ('Occipital', occipital)]:
        stability = df[channels].std(axis=1).rolling(window=50).mean()
        plt.plot(stability[:1000], label=region)
    plt.title('Signal Stability Analysis by Region')
    plt.legend()
    plt.ylabel('Rolling Standard Deviation')
    plt.show()

    # 15. Channel Energy Distribution
    energy = df.apply(lambda x: np.sum(x**2))
    plt.figure(figsize=(12, 6))
    energy.plot(kind='pie', autopct='%1.1f%%')
    plt.title('Channel Energy Distribution')
    plt.show()

    # 16. Temporal Evolution Heatmap
    plt.figure(figsize=(15, 8))
    sns.heatmap(df[eeg_channels].iloc[:1000].T, 
                cmap='seismic', center=0)
    plt.title('Temporal Evolution of Channel Activities')
    plt.xlabel('Time Points')
    plt.ylabel('Channels')
    plt.show()

    # 17. Channel Amplitude Range Analysis
    ranges = df.apply(lambda x: x.max() - x.min())
    plt.figure(figsize=(12, 6))
    ranges.plot(kind='bar')
    plt.title('Channel Amplitude Ranges')
    plt.xticks(rotation=45)
    plt.ylabel('Amplitude Range (μV)')
    plt.show()

    # 18. Regional Coherence Plot
    coherence_data = []
    for region in ['Frontal', 'Central', 'Parietal', 'Temporal', 'Occipital']:
        if region == 'Frontal':
            channels = frontal
        elif region == 'Central':
            channels = central
        elif region == 'Parietal':
            channels = parietal
        elif region == 'Temporal':
            channels = temporal
        else:
            channels = occipital
            
        regional_coherence = df[channels].corr().mean().mean()
        coherence_data.append((region, regional_coherence))
    
    coherence_df = pd.DataFrame(coherence_data, columns=['Region', 'Coherence'])
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Region', y='Coherence', data=coherence_df)
    plt.title('Regional Signal Coherence')
    plt.xticks(rotation=45)
    plt.show()

    # 19. Multi-Channel Phase Analysis
    plt.figure(figsize=(15, 8))
    for channel in ['F3', 'F4', 'C3', 'C4']:
        analytic_signal = signal.hilbert(df[channel])
        phase = np.angle(analytic_signal)
        plt.plot(phase[:500], label=channel)
    plt.title('Phase Analysis of Selected Channels')
    plt.legend()
    plt.ylabel('Phase (radians)')
    plt.show()

    # 20. Statistical Moments Plot
    moments_df = pd.DataFrame({
        'Mean': df.mean(),
        'Std': df.std(),
        'Skewness': df.apply(skew),
        'Kurtosis': df.apply(kurtosis)
    })
    
    plt.figure(figsize=(15, 8))
    moments_df.plot(kind='bar', subplots=True, layout=(2,2), figsize=(15,10))
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    create_eeg_visualizations(df)
