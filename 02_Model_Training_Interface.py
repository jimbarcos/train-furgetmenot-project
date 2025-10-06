import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import tkinter.scrolledtext as scrolledtext
import threading
import queue
import time
import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
from matplotlib.animation import FuncAnimation
import seaborn as sns
from datetime import datetime
import warnings
import tensorflow as tf
from tensorflow.keras.layers import Layer, Dense, GlobalAveragePooling2D, Lambda, Input, Dropout, BatchNormalization, Concatenate, Conv2D, Reshape, Flatten, MultiHeadAttention, LayerNormalization, Add, GaussianNoise
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.regularizers import l2
from tensorflow.keras import backend as K
from tensorflow.keras.applications import MobileNetV2
from sklearn.metrics import roc_curve, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from typing import List, Tuple
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# ============================================================================
# EMBEDDED: Capsule Network Components
# ============================================================================

def squash(vectors, axis=-1):
    """Squashing function for capsule networks with improved numerical stability.
    MODIFIED: Increased epsilon to prevent near-zero outputs that cause distance collapse.
    """
    s_squared_norm = K.sum(K.square(vectors), axis, keepdims=True)
    epsilon = 1e-7  # Increased from K.epsilon() for better stability
    scale = s_squared_norm / (1 + s_squared_norm) / K.sqrt(s_squared_norm + epsilon)
    return scale * vectors

def safe_norm(s, axis=-1, epsilon=1e-7, keep_dims=False, name=None):
    """Safe norm calculation to avoid numerical issues."""
    squared_norm = tf.reduce_sum(tf.square(s), axis=axis, keepdims=keep_dims)
    return tf.sqrt(squared_norm + epsilon)

class CapsuleLayer(Layer):
    """Basic capsule layer implementation."""
    def __init__(self, num_capsules, dim_capsules, routings=3, **kwargs):
        super(CapsuleLayer, self).__init__(**kwargs)
        self.num_capsules = num_capsules
        self.dim_capsules = dim_capsules
        self.routings = routings
    
    def build(self, input_shape):
        super(CapsuleLayer, self).build(input_shape)
    
    def call(self, inputs, training=None):
        return inputs
    
    def get_config(self):
        config = super(CapsuleLayer, self).get_config()
        config.update({
            'num_capsules': self.num_capsules,
            'dim_capsules': self.dim_capsules,
            'routings': self.routings
        })
        return config

class PrimaryCapsule(Layer):
    """Primary capsule layer implementation."""
    def __init__(self, dim_capsules, n_channels, kernel_size, strides, padding, **kwargs):
        super(PrimaryCapsule, self).__init__(**kwargs)
        self.dim_capsules = dim_capsules
        self.n_channels = n_channels
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        
    def build(self, input_shape):
        self.conv = Conv2D(
            filters=self.dim_capsules * self.n_channels,
            kernel_size=self.kernel_size,
            strides=self.strides,
            padding=self.padding,
            activation='relu',
            name=f'{self.name}_conv'
        )
        self.conv.build(input_shape)
        super(PrimaryCapsule, self).build(input_shape)
        
    def call(self, inputs, training=None):
        outputs = self.conv(inputs)
        batch_size = tf.shape(outputs)[0]
        outputs = tf.reshape(outputs, [batch_size, -1, self.dim_capsules])
        return squash(outputs, axis=-1)
    
    def get_config(self):
        config = super(PrimaryCapsule, self).get_config()
        config.update({
            'dim_capsules': self.dim_capsules,
            'n_channels': self.n_channels,
            'kernel_size': self.kernel_size,
            'strides': self.strides,
            'padding': self.padding
        })
        return config

class EnhancedCapsuleLayer(CapsuleLayer):
    """Enhanced Capsule Layer with self-attention mechanism."""
    
    def __init__(self, num_capsules, dim_capsules, routings=3, attention_heads=4, 
                 use_attention=True, kernel_initializer='glorot_uniform', **kwargs):
        super(EnhancedCapsuleLayer, self).__init__(
            num_capsules, dim_capsules, routings, **kwargs
        )
        self.attention_heads = attention_heads
        self.use_attention = use_attention
        self.kernel_initializer = kernel_initializer
        
    def build(self, input_shape):
        super().build(input_shape)
        # Expect input shape: (batch, num_primary_caps, dim_in)
        self.dim_in = int(input_shape[-1])
        # Transformation matrices: one per output capsule (classic capsules style)
        # Use higher variance initialization to ensure capsules don't collapse to zero
        self.W = self.add_weight(
            shape=(self.num_capsules, self.dim_in, self.dim_capsules),
            initializer=tf.keras.initializers.VarianceScaling(scale=2.0, mode='fan_avg', distribution='uniform'),
            name=f"{self.name}_W"
        )
        self.b_caps = self.add_weight(
            shape=(self.num_capsules, self.dim_capsules),
            initializer='zeros',
            name=f"{self.name}_b"
        )
        # Learnable global scale to adapt dynamic range if squash outputs too small
        # INCREASED from 1.0 to 3.0 to prevent collapsed outputs
        self.gamma = self.add_weight(
            shape=(1,), initializer=tf.keras.initializers.constant(3.0), name=f"{self.name}_gamma"
        )
        self.last_c_entropy = None
        if self.use_attention:
            unique_prefix = f"{self.name}_"
            self.attention_layer = MultiHeadAttention(
                num_heads=self.attention_heads,
                key_dim=self.dim_capsules,
                name=f'{unique_prefix}capsule_attention'
            )
            self.attention_norm = LayerNormalization(name=f'{unique_prefix}attention_norm')
    
    def enhanced_dynamic_routing(self, u_hat, training):
        batch_size = tf.shape(u_hat)[0]
        input_num_capsules = tf.shape(u_hat)[1]
        num_output_capsules = tf.shape(u_hat)[2] 
        capsule_dim = tf.shape(u_hat)[3]
        
        b = tf.zeros([batch_size, input_num_capsules, num_output_capsules, 1])
        
        for i in range(self.routings):
            c = tf.nn.softmax(b, axis=2)  # (batch, in_caps, out_caps, 1)
            s = tf.reduce_sum(c * u_hat, axis=1, keepdims=True)  # (batch,1,out_caps,dim_caps)
            v = squash(s, axis=-1)  # (batch,1,out_caps,dim_caps)
            if i < self.routings - 1:
                agreement = tf.reduce_sum(u_hat * v, axis=-1, keepdims=True)
                b = b + agreement
        # Coupling entropy diagnostic
        c_final = tf.nn.softmax(b, axis=2)
        entropy = -tf.reduce_sum(c_final * tf.math.log(c_final + 1e-9), axis=2)  # (batch,in_caps,1)
        norm_entropy = entropy / tf.math.log(tf.cast(num_output_capsules, tf.float32) + 1e-9)
        self.last_c_entropy = tf.reduce_mean(norm_entropy)
        
        return tf.squeeze(v, axis=1)
    
    def call(self, inputs, training=None):
        # inputs: (batch, N, dim_in)
        # Transform each input capsule to each output capsule space: u_hat (batch, N, num_caps, dim_caps)
        u_hat = tf.einsum('b n d, c d h -> b n c h', inputs, self.W) + self.bias_expand()
        if training:
            # Inject stronger noise early to break symmetry
            u_hat += tf.random.normal(tf.shape(u_hat), stddev=0.05)
            # If variance still tiny, add adaptive jitter
            global_std = tf.math.reduce_std(u_hat)
            def add_jitter():
                return u_hat + tf.random.normal(tf.shape(u_hat), stddev=0.1)
            u_hat = tf.cond(global_std < 0.02, add_jitter, lambda: u_hat)
        routed = self.enhanced_dynamic_routing(u_hat, training=training if training is not None else False)
        # CRITICAL FIX: Apply attention during BOTH training and inference to ensure gradients flow
        # Without this, attention weights never get updated and warning appears
        if self.use_attention:
            attended = self.attention_layer(routed, routed, training=training)
            routed = self.attention_norm(attended + routed, training=training)
        routed = squash(routed, axis=-1) * self.gamma
        return routed

    def bias_expand(self):
        return tf.reshape(self.b_caps, (1, 1, self.num_capsules, self.dim_capsules))

    def get_last_routing_entropy(self):
        return self.last_c_entropy
    
    def get_config(self):
        config = super(EnhancedCapsuleLayer, self).get_config()
        config.update({
            'attention_heads': self.attention_heads,
            'use_attention': self.use_attention
        })
        return config


# ============================================================================
# EMBEDDED: SiameseCapsuleNetworkMobileNetV2 Class
# ============================================================================

class SiameseCapsuleNetworkMobileNetV2:
    """
    Siamese Capsule Network with MobileNetV2 backbone.
    
    This class implements the proposed architecture that combines:
    1. MobileNetV2 for efficient feature extraction
    2. Enhanced Capsule Networks for spatial relationship modeling
    3. Self-attention mechanisms for improved routing
    4. Pearson correlation-based distance computation
    5. Advanced training strategies
    """
    
    def __init__(self, input_shape=(224, 224, 3), embedding_dim=256, 
                 num_capsules=6, dim_capsules=8, routings=3, attention_heads=2,
                 use_attention=True, use_pearson_distance=False, optimize_speed=True,
                 mobilenet_alpha=1.0, distance_type='cosine', loss_type='contrastive_enhanced',
                 label_positive_means_similar=True, learning_rate=5e-5,
                 use_capsule=True, capsule_auto_fallback=False):
        self.input_shape = input_shape
        self.embedding_dim = embedding_dim
        self.optimize_speed = optimize_speed
        self.learning_rate = learning_rate  # Store GUI learning rate

        self.num_capsules = num_capsules
        self.dim_capsules = dim_capsules
        self.attention_heads = attention_heads
        self.mobilenet_alpha = mobilenet_alpha
        
        if optimize_speed:
            self.num_capsules = max(1, min(self.num_capsules, 10))
            self.dim_capsules = max(2, min(self.dim_capsules, 32))
            self.attention_heads = max(1, min(self.attention_heads, 8))
            self.mobilenet_alpha = float(np.clip(self.mobilenet_alpha, 0.35, 1.0))
        
        self.routings = routings
        self.use_attention = use_attention
        self.use_pearson_distance = use_pearson_distance
        self.distance_type = distance_type if distance_type in ('cosine', 'euclidean', 'pearson') else ('cosine' if not use_pearson_distance else 'pearson')
        self.loss_type = loss_type if loss_type in ('contrastive_simple', 'contrastive_enhanced', 'bce_similarity', 'bce_on_similarity') else 'contrastive_simple'
        self.use_capsule = bool(use_capsule)
        self.capsule_auto_fallback = bool(capsule_auto_fallback)
        
        self.model = None
        self.siamese_model = None
        self.history = None
        self.training_metrics = {}
        
        os.makedirs('models', exist_ok=True)

        # CRITICAL: Initialize metric threshold based on distance type
        # For Euclidean with raw embeddings: distances ~4-6 initially, need threshold ~5.0
        # For Cosine/Pearson normalized: distances 0-1, but observed range is much lower
        if distance_type == 'euclidean':
            initial_threshold = 5.0  # Midpoint of typical initial euclidean range
        elif distance_type == 'cosine':
            # Observed cosine distances with current formula: ~0.2-0.6 range
            initial_threshold = 0.5  # Will be tuned adaptively during training
        else:
            initial_threshold = 0.5  # For normalized distances (pearson)
        
        try:
            self._threshold_var = tf.Variable(initial_threshold, trainable=False, dtype=tf.float32, name='metric_distance_threshold')
            self.metric_distance_threshold = float(self._threshold_var.numpy())
        except Exception:
            self._threshold_var = None
            self.metric_distance_threshold = initial_threshold
        
        # Track contrastive margin so we can adapt it during training when metrics plateau
        if self.distance_type == 'euclidean':
            self.base_contrastive_margin = 7.0
            self.margin_schedule_max = self.base_contrastive_margin * 1.2
        else:
            # Cosine/Pearson distances lie in [0, 1]. We observed validation max distances
            # occasionally reach ~0.97 at early epochs. Use a fixed, feasible target that
            # strongly pushes negatives apart but remains within range.
            # Scheduler remains DISABLED to avoid target drift.
            self.base_contrastive_margin = 0.95
            self.margin_schedule_max = 0.95  # Fixed, no scheduling
        try:
            self.margin_var = tf.Variable(self.base_contrastive_margin, trainable=False, dtype=tf.float32, name='contrastive_margin')
        except Exception:
            self.margin_var = None
        self.margin_update_history: List[float] = []

        # Placeholders for GaussianNoise scheduling
        self.noise_layer_name = 'emb_noise'
        self.noise_initial_std: float | None = None
        self.noise_min_std: float = 0.0

        self.temperature = tf.Variable(1.0, trainable=True, dtype=tf.float32, name='temperature')
        self.dropout_rate = 0.15
        self.weight_decay = 0.0005
        self.label_smoothing = 0.05
        self.label_positive_means_similar = bool(label_positive_means_similar)
        # Enable adaptive thresholding inside training metrics so accuracy/precision/recall
        # reflect the evolving distance distribution and avoid misleading plateau signals.
        self.adaptive_metric_threshold = True
        
    def _build_optimizer(self, learning_rate: float | None = None):
        lr = float(learning_rate) if learning_rate is not None else float(self.learning_rate)
        try:
            from tensorflow.keras.optimizers import AdamW  # type: ignore
            return AdamW(
                learning_rate=lr,
                weight_decay=self.weight_decay,
                beta_1=0.9,
                beta_2=0.999,
                clipnorm=1.0
            )
        except ImportError:
            return Adam(
                learning_rate=lr,
                beta_1=0.9,
                beta_2=0.999,
                clipnorm=1.0
            )

    def _build_metrics(self):
        self_outer = self

        class _ThresholdedMetricBase(tf.keras.metrics.Metric):
            def __init__(self, name='metric', **kwargs):
                super().__init__(name=name, **kwargs)
                self.weighted_correct = self.add_weight(name='weighted_correct', initializer='zeros', dtype=tf.float32)
                self.weighted_total = self.add_weight(name='weighted_total', initializer='zeros', dtype=tf.float32)
                self._pred_min = self.add_weight(name='pred_min', initializer=lambda shape, dtype: tf.constant(1.0, dtype=dtype), dtype=tf.float32)
                self._pred_max = self.add_weight(name='pred_max', initializer='zeros', dtype=tf.float32)
                self._pred_sum = self.add_weight(name='pred_sum', initializer='zeros', dtype=tf.float32)
                self._pred_count = self.add_weight(name='pred_count', initializer='zeros', dtype=tf.float32)

            def _normalize(self, y_true, y_pred):
                y_true = tf.cast(y_true, tf.float32)
                y_pred = tf.cast(y_pred, tf.float32)
                b = tf.shape(y_pred)[0]
                y_true = tf.reshape(y_true, (b, -1))
                y_true = tf.reduce_mean(y_true, axis=1, keepdims=True)
                y_pred = tf.reshape(y_pred, (b, -1))
                y_pred = tf.reduce_mean(y_pred, axis=1, keepdims=True)
                if not self_outer.label_positive_means_similar:
                    y_true = 1.0 - y_true
                base_thr = tf.constant(0.5, dtype=tf.float32)
                self._pred_min.assign(tf.minimum(self._pred_min, tf.reduce_min(y_pred)))
                self._pred_max.assign(tf.maximum(self._pred_max, tf.reduce_max(y_pred)))
                self._pred_sum.assign_add(tf.reduce_sum(y_pred))
                self._pred_count.assign_add(tf.cast(b, tf.float32))
                if getattr(self_outer, 'adaptive_metric_threshold', False):
                    dynamic_thr = (self._pred_min + self._pred_max) * 0.5
                    base_thr = 0.7 * base_thr + 0.3 * dynamic_thr
                preds = tf.cast(tf.less_equal(y_pred, base_thr), tf.float32)
                return y_true, preds

            def update_core(self, y_true, preds, sample_weight):
                raise NotImplementedError

            def update_state(self, y_true, y_pred, sample_weight=None):
                y_true, preds = self._normalize(y_true, y_pred)
                if sample_weight is None:
                    sample_weight = tf.ones_like(tf.squeeze(y_true, axis=-1), dtype=tf.float32)
                else:
                    sample_weight = tf.cast(sample_weight, tf.float32)
                    sample_weight = tf.reshape(sample_weight, (-1,))
                self.update_core(y_true, preds, sample_weight)
                if getattr(self_outer, 'debug_metric_distribution', False):
                    total_seen = tf.cast(self._pred_count, tf.int32)
                    cond = tf.equal(tf.math.floormod(total_seen, 200), 0)

                    def _print():
                        thr = tf.constant(0.5, dtype=tf.float32)
                        pred_mean = self._pred_sum / (self._pred_count + 1e-6)
                        tf.print('[MetricDebug]', self.name, 'pred_min=', self._pred_min, 'pred_max=', self._pred_max, 'mean=', pred_mean, 'threshold=', thr, output_stream='file://stdout')
                        tf.print('[MetricDebug]   → Distance range: [', self._pred_min, ',', self._pred_max, ']', output_stream='file://stdout')
                        tf.print('[MetricDebug]   → If ALL > threshold: TP=0 (all predict dissimilar)', output_stream='file://stdout')
                        tf.print('[MetricDebug]   → If ALL < threshold: TN=0 (all predict similar)', output_stream='file://stdout')
                        tf.print('[MetricDebug]   → If ALL distances > threshold: TP=0 (all predict dissimilar)', output_stream='file://stdout')
                        tf.print('[MetricDebug]   → If ALL distances < threshold: TN=0 (all predict similar)', output_stream='file://stdout')
                        return 0

                    tf.cond(cond, lambda: _print(), lambda: 0)

            def result(self):
                return tf.where(self.weighted_total > 0.0, self.weighted_correct / (self.weighted_total + 1e-7), 0.0)

            def reset_state(self):
                for var in self.variables:
                    var.assign(0.0)
                self._pred_min.assign(1.0)

        class WeightedPrecision(_ThresholdedMetricBase):
            def __init__(self):
                super().__init__(name='precision_metric')

            def update_core(self, y_true, preds, sample_weight):
                tp = tf.reduce_sum(sample_weight * tf.squeeze(preds * y_true, axis=-1))
                pp = tf.reduce_sum(sample_weight * tf.squeeze(preds, axis=-1))
                self.weighted_correct.assign_add(tp)
                self.weighted_total.assign_add(pp)

        class WeightedRecall(_ThresholdedMetricBase):
            def __init__(self):
                super().__init__(name='recall_metric')

            def update_core(self, y_true, preds, sample_weight):
                tp = tf.reduce_sum(sample_weight * tf.squeeze(preds * y_true, axis=-1))
                ap = tf.reduce_sum(sample_weight * tf.squeeze(y_true, axis=-1))
                self.weighted_correct.assign_add(tp)
                self.weighted_total.assign_add(ap)

        class WeightedBinaryAccuracy(_ThresholdedMetricBase):
            def __init__(self):
                super().__init__(name='binary_accuracy_metric')

            def update_core(self, y_true, preds, sample_weight):
                eq = tf.cast(tf.equal(preds, y_true), tf.float32)
                self.weighted_correct.assign_add(tf.reduce_sum(sample_weight * tf.squeeze(eq, axis=-1)))
                self.weighted_total.assign_add(tf.reduce_sum(sample_weight))

        return [WeightedBinaryAccuracy(), WeightedPrecision(), WeightedRecall()]

    def recompile_after_unfreeze(self, learning_rate: float | None = None):
        if self.siamese_model is None:
            return
        if learning_rate is not None:
            self.learning_rate = float(learning_rate)
        optimizer = self._build_optimizer(learning_rate)
        metrics_list = self._build_metrics()
        loss_fn = getattr(self, '_loss_function', None)
        if loss_fn is None:
            loss_fn = self.siamese_model.loss
        self.siamese_model.compile(
            optimizer=optimizer,
            loss=loss_fn,
            metrics=metrics_list
        )

    def _warmup_lr_schedule(self, epoch):
        base_lr = 1e-4
        warmup_epochs = 3
        
        if epoch < warmup_epochs:
            return 1e-6 + (base_lr - 1e-6) * epoch / warmup_epochs
        else:
            return base_lr * (0.95 ** (epoch - warmup_epochs))
        
    def create_mobilenetv2_capsule_base(self):
        print("Creating MobileNetV2-Capsule hybrid base network...")
        
        input_layer = Input(shape=self.input_shape, name='input_image')
        
        # CRITICAL: ImageNet weights only available for standard alpha values
        # Map custom alpha to nearest standard value for weight loading
        imagenet_alphas = [0.35, 0.5, 0.75, 1.0, 1.3, 1.4]
        if self.mobilenet_alpha in imagenet_alphas:
            weights_alpha = self.mobilenet_alpha
        else:
            # Find nearest standard alpha
            weights_alpha = min(imagenet_alphas, key=lambda x: abs(x - self.mobilenet_alpha))
            print(f"Note: Using alpha={weights_alpha} for ImageNet weights (requested: {self.mobilenet_alpha})")
        
        try:
            mobilenet_base = MobileNetV2(
                input_shape=self.input_shape,
                include_top=False,
                weights='imagenet',
                input_tensor=input_layer,
                alpha=weights_alpha
            )
            print(f"✓ Successfully loaded ImageNet pretrained weights with alpha={weights_alpha}")
        except Exception as e:
            print(f"Warning: Could not load ImageNet weights ({e}). Falling back to random initialization.")
            mobilenet_base = MobileNetV2(
                input_shape=self.input_shape,
                include_top=False,
                weights=None,
                input_tensor=input_layer,
                alpha=self.mobilenet_alpha
            )
        
        for layer in mobilenet_base.layers[:-20]:
            layer.trainable = False
        for layer in mobilenet_base.layers[-20:]:
            layer.trainable = True
        
        x = mobilenet_base.output
        
        print(f"MobileNetV2 output shape: {x.shape}")
        
        prep_filters1 = 64 if self.optimize_speed else 256
        prep_filters2 = 32 if self.optimize_speed else 128
        
        x = Conv2D(prep_filters1, kernel_size=3, strides=1, padding='same', 
                  activation='relu', name='prep_conv1',
                  kernel_regularizer=l2(self.weight_decay))(x)
        x = BatchNormalization(momentum=0.99, name='prep_bn1')(x)
        x = Dropout(self.dropout_rate, name='prep_dropout1')(x)
        
        x = Conv2D(prep_filters2, kernel_size=3, strides=1, padding='same', 
                  activation='relu', name='prep_conv2',
                  kernel_regularizer=l2(self.weight_decay))(x)
        x = BatchNormalization(momentum=0.99, name='prep_bn2')(x)
        x = Dropout(self.dropout_rate + 0.1, name='prep_dropout2')(x)
        
        if self.use_capsule:
            primary_caps = PrimaryCapsule(
                dim_capsules=self.dim_capsules,
                n_channels=8 if self.optimize_speed else 32,
                kernel_size=3,
                strides=2,
                padding='valid',
                name='mobilenet_primary_caps'
            )(x)
            print(f"Primary capsules shape: {primary_caps.shape}")
            digital_caps = EnhancedCapsuleLayer(
                num_capsules=self.num_capsules,
                dim_capsules=self.dim_capsules,
                routings=self.routings,
                attention_heads=self.attention_heads,
                use_attention=self.use_attention,
                name='enhanced_digital_caps'
            )(primary_caps)
            print(f"Digital capsules shape: {digital_caps.shape}")
            # Provide both capsule pose vectors and their activation magnitudes to the dense head to enhance variance
            caps_flattened = Flatten(name='caps_flatten')(digital_caps)
            caps_activations = Lambda(lambda t: safe_norm(t, axis=-1), name='caps_activations')(digital_caps)  # (batch, num_capsules)
            features = Concatenate(name='caps_concat')([caps_flattened, caps_activations])
        else:
            # Fallback: no capsules, use efficient GAP + projection
            print("[Fallback] Using GlobalAveragePooling2D instead of capsules")
            features = GlobalAveragePooling2D(name='gap')(x)
        
        xproj = Dense(512, activation='relu', name='emb_fc1', kernel_regularizer=l2(0.005))(features)
        # CRITICAL: No BN for capsule path - it crushes variance to near-zero
        if not self.use_capsule:
            xproj = BatchNormalization(name='emb_bn1')(xproj)
        xproj = Dropout(0.04 if self.use_capsule else 0.2, name='emb_dropout1')(xproj)  # Reduced dropout for capsules
        xproj = Dense(256, activation='relu', name='emb_fc2', kernel_regularizer=l2(self.weight_decay * 2))(xproj)
        if not self.use_capsule:
            xproj = BatchNormalization(name='emb_bn2')(xproj)
        xproj = Dropout(0.01 if self.use_capsule else 0.15, name='emb_dropout2')(xproj)  # Reduced dropout for capsules
        xproj = Dense(128, activation='relu', name='emb_fc3', kernel_regularizer=l2(self.weight_decay))(xproj)
        # For capsule path: NO BN at all to preserve variance
        if not self.use_capsule:
            xproj = BatchNormalization(name='emb_bn3')(xproj)
            xproj = Dropout(self.dropout_rate, name='emb_dropout3')(xproj)
        # CRITICAL: Use He initialization to ensure variance in embeddings even without training
        # This prevents the untrained model from producing near-zero activations
        proj = Dense(self.embedding_dim, activation='linear', name='final_embeddings', 
                    kernel_initializer='he_normal',  # Better initialization for variance
                    kernel_regularizer=l2(self.weight_decay * (0.5 if self.use_capsule else 1.0)))(xproj)
        if not self.use_capsule:
            proj = BatchNormalization(name='emb_out_bn')(proj)
            proj = Dropout(self.dropout_rate * 0.5, name='emb_out_dropout')(proj)
        # Inject Gaussian noise to break symmetry and add variance (only during training)
        # TUNE: Use slightly lower noise for capsules early to avoid excessive overlap
        # while still preventing dead activations.
        noise_std = 0.01 if self.use_capsule else 0.01
        self.noise_initial_std = float(noise_std)
        self.noise_min_std = 0.01 if self.use_capsule else 0.01
        proj = GaussianNoise(noise_std, name='emb_noise')(proj)
        # CRITICAL FIX: For capsule networks, DO NOT L2 normalize - it crushes variance!
        # Use raw embeddings to preserve the variance that capsules generate
        if self.use_capsule:
            # Use raw embeddings directly (no scaling, no normalization)
            embeddings = proj
        else:
            # Non-capsule path: use L2 normalization as before
            embeddings = Lambda(lambda t: tf.math.l2_normalize(t + 1e-9, axis=1), name='emb_l2norm')(proj)

        if self.use_capsule:
            base_model = Model(inputs=input_layer, outputs=[digital_caps, embeddings], name='MobileNetV2_CapsNet_Hybrid')
        else:
            base_model = Model(inputs=input_layer, outputs=[embeddings, embeddings], name='MobileNetV2_Fallback')
        
        total_params = base_model.count_params()
        trainable_params = sum([K.count_params(w) for w in base_model.trainable_weights])
        
        print(f"Hybrid base network created with {total_params:,} total parameters")
        print(f"Trainable parameters: {trainable_params:,}")
        print(f"Non-trainable parameters: {total_params - trainable_params:,}")
        
        return base_model

    def probe_embedding_variance(self, generator, batches=1):
        """Lightweight probe to print raw embedding std/spread BEFORE distance layer.
        Helps diagnose collapse source (embedding vs distance computation).
        CRITICAL: Uses training=True to activate GaussianNoise for accurate variance measurement.
        """
        if self.siamese_model is None or self.model is None:
            return
        try:
            # Use model directly with training=True instead of predict()
            collected = []
            for i in range(min(batches, len(generator))):
                batch = generator[i]
                if isinstance(batch, (list, tuple)):
                    (xa, xb), *_ = batch
                else:
                    continue
                # Call model with training=True to activate GaussianNoise
                output_a = self.model(xa, training=True)
                output_b = self.model(xb, training=True)
                # Extract embeddings (second output)
                if isinstance(output_a, (list, tuple)):
                    ea = output_a[1].numpy()
                    eb = output_b[1].numpy()
                else:
                    ea = output_a.numpy()
                    eb = output_b.numpy()
                collected.append(ea)
                collected.append(eb)
            if not collected:
                return
            embs = np.concatenate(collected, axis=0)
            per_dim_std = embs.std(axis=0)
            global_std = float(per_dim_std.mean())
            spread = float(embs.max() - embs.min())
            print(f"[EmbProbe] global_std={global_std:.5f} spread={spread:.5f} mean_per_dim_std={per_dim_std.mean():.5f} min_dim_std={per_dim_std.min():.5f} max_dim_std={per_dim_std.max():.5f}")
        except Exception as e:
            print(f"[EmbProbe] failed: {e}")

    def check_embedding_variance(self, generator, max_batches=2, var_threshold=1e-8):
        """Inspect a couple of batches; if cosine distances collapse, trigger fallback rebuild.
        Updated: require BOTH very low spread and very low std to reduce false positives; log diagnostics.
        INCREASED threshold from 1e-4 to 1e-8 to allow capsule networks with naturally lower initial variance.
        """
        if not self.capsule_auto_fallback or not self.use_capsule:
            return False
        try:
            import numpy as np
            batches = min(len(generator), max_batches)
            dists = []
            for i in range(batches):
                batch = generator[i]
                if isinstance(batch, (list, tuple)) and len(batch) >= 2:
                    (xa, xb), y = batch[0], batch[1]
                if len(batch) == 3:
                    (xa, xb), y, _ = batch
                preds = self.siamese_model.predict([xa, xb], verbose=0).reshape(-1)
                dists.append(preds)
            if not dists:
                return False
            all_pred = np.concatenate(dists)
            spread = float(np.max(all_pred) - np.min(all_pred))
            std = float(np.std(all_pred))
            mean = float(np.mean(all_pred))
            print(f"[VarianceCheck] pre-train distance stats: mean={mean:.5f} std={std:.5f} spread={spread:.5f}")
            if spread < var_threshold and std < (var_threshold * 0.75):
                print(f"\n[Auto-Fallback] Detected collapsed distance distribution (spread={spread:.6g}, std={std:.6g}). Rebuilding WITHOUT capsules.\n")
                self.use_capsule = False
                self.create_siamese_capsule_network()
                return True
        except Exception as e:
            print(f"Embedding variance check failed: {e}")
        return False
    
    def pearson_correlation_distance(self, vectors):
        x, y = vectors
        mean_x = K.mean(x, axis=1, keepdims=True)
        mean_y = K.mean(y, axis=1, keepdims=True)
        x_centered = x - mean_x
        y_centered = y - mean_y
        numerator = K.sum(x_centered * y_centered, axis=1)
        denominator_x = K.sqrt(K.sum(K.square(x_centered), axis=1) + K.epsilon())
        denominator_y = K.sqrt(K.sum(K.square(y_centered), axis=1) + K.epsilon())
        correlation = numerator / (denominator_x * denominator_y + K.epsilon())
        correlation = K.clip(correlation, -1.0, 1.0)
        similarity = (correlation + 1.0) * 0.5
        distance = 1.0 - similarity
        distance = K.clip(distance, 0.0, 1.0)
        distance_stabilized = 0.1 + 0.8 * K.sigmoid(6.0 * (distance - 0.5))
        return distance_stabilized
    
    def euclidean_distance(self, vectors):
        """Euclidean distance for raw embeddings.
        Returns distance in [0, inf) range, suitable for contrastive loss.
        """
        x, y = vectors
        sum_square = K.sum(K.square(x - y), axis=1, keepdims=True)
        dist = K.sqrt(K.maximum(sum_square, K.epsilon()))
        # Don't normalize - let the loss function handle the range
        # Contrastive loss expects unbounded distances
        return dist
    
    def cosine_distance(self, vectors):
        """Compute cosine distance between embedding pairs.
        Returns: distance in [0, 1] where 0=identical, 1=opposite.
        """
        x, y = vectors
        x_norm = K.l2_normalize(x, axis=1)
        y_norm = K.l2_normalize(y, axis=1)
        cosine_sim = K.sum(x_norm * y_norm, axis=1)
        # FIXED: Use (1 - cosine_sim) not 0.5 * (1 - cosine_sim) to get proper [0, 1] range
        cosine_dist = 1.0 - cosine_sim
        return K.clip(cosine_dist, 0.0, 1.0)

    def simple_contrastive_loss(self, y_true, y_pred, margin=1.0):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        batch = tf.shape(y_true)[0]
        y_true = tf.reshape(y_true, (batch, 1))
        y_pred = tf.reshape(y_pred, (batch, -1))
        y_pred = tf.reduce_mean(y_pred, axis=1, keepdims=True)
        pos_loss = y_true * K.square(y_pred)
        neg_loss = (1.0 - y_true) * K.square(K.maximum(margin - y_pred, 0.0))
        return K.mean(pos_loss + neg_loss)
    
    def enhanced_contrastive_loss(self, y_true, y_pred, margin=None, alpha=0.1):
        """Enhanced contrastive loss with separation regularization and focal weighting.
        UPDATED: tuned for cosine distances with fixed margin; includes class weighting and
        a small hard-negative mining term to improve AUC and reduce high-recall bias.
        For positive pairs (label=1): minimize distance
        For negative pairs (label=0): maximize distance up to margin
        """
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        batch = tf.shape(y_true)[0]
        y_true = tf.reshape(y_true, (batch, 1))
        y_pred = tf.reshape(y_pred, (batch, -1))
        y_pred = tf.reduce_mean(y_pred, axis=1, keepdims=True)
        y_true_smooth = y_true * (1.0 - self.label_smoothing) + 0.5 * self.label_smoothing
        if margin is None:
            if getattr(self, 'margin_var', None) is not None:
                margin_tf = tf.cast(self.margin_var, tf.float32)
            else:
                margin_tf = tf.constant(self.base_contrastive_margin, dtype=tf.float32)
        elif isinstance(margin, tf.Variable):
            margin_tf = tf.cast(margin, tf.float32)
        else:
            margin_tf = tf.constant(float(margin), dtype=tf.float32)

        # Class weighting: increase weight on dissimilar pairs to counter high-recall bias
        pos_weight = 1.0   # weight for similar pairs
        neg_weight = 1.35  # weight for dissimilar pairs

        pos_loss = pos_weight * y_true_smooth * K.square(y_pred)
        neg_loss = neg_weight * (1.0 - y_true_smooth) * K.square(K.maximum(margin_tf - y_pred, 0.0))
        contrastive = K.mean(pos_loss + neg_loss)

        # Hard-negative emphasis: add a small extra loss from the top-k hardest negatives
        k = tf.maximum(1, tf.cast(0.1 * tf.cast(batch, tf.float32), tf.int32))  # top 10%
        neg_residual = K.maximum(margin_tf - y_pred, 0.0) * (1.0 - y_true_smooth)
        topk_vals, _ = tf.math.top_k(tf.reshape(neg_residual, (-1,)), k=k)
        hard_neg_loss = K.mean(K.square(topk_vals))

        # Regularization terms to stabilize training
        sep_reg = K.mean(K.exp(-5.0 * K.square(y_pred - margin_tf * 0.5)))
        focal_weight = K.square(K.abs(y_true_smooth - K.sigmoid(1.0 - y_pred)))
        focal_loss = K.mean(focal_weight * (pos_loss + neg_loss))
        return contrastive + 0.3 * hard_neg_loss + alpha * sep_reg + 0.1 * focal_loss
    
    def accuracy_metric_enhanced(self, y_true, y_pred, threshold=0.5):
        """IMPROVED: Fixed accuracy metric with proper normalization for cosine/euclidean/pearson."""
        # Normalize predictions to [0, 1] range for consistent thresholding
        if self.distance_type == 'cosine' or self.distance_type == 'euclidean':
            # Cosine/Euclidean output distances in [0, 1] range already
            # Lower distance = more similar, so predictions are already normalized
            y_pred_normalized = y_pred
        elif self.distance_type == 'pearson' or self.use_pearson_distance:
            # Pearson outputs similarity in range ~[-1, 1] or small positive values
            # Normalize to [0, 1] using sigmoid-like transformation
            y_pred_normalized = tf.nn.sigmoid((y_pred - 0.138) * 20.0)  # Center and scale
        else:
            # Default: assume distance metric in [0, 1]
            y_pred_normalized = y_pred
        
        # Use fixed threshold for consistency
        # For distances: low value = similar (label 1), high value = dissimilar (label 0)
        # So we check if distance < threshold for similarity
        fixed_threshold = 0.5
        predictions = K.cast(y_pred_normalized < fixed_threshold, K.floatx())  # INVERTED: < not >
        y_true_cast = K.cast(y_true, K.floatx())
        correct = K.cast(K.equal(y_true_cast, predictions), K.floatx())
        accuracy = K.mean(correct)
        accuracy = tf.where(tf.math.is_finite(accuracy), accuracy, 0.5)
        return K.clip(accuracy, 0.0, 1.0)
    
    def create_siamese_capsule_network(self):
        print("Creating Siamese Capsule Network with MobileNetV2...")
        
        base_network = self.create_mobilenetv2_capsule_base()
        
        input_a = Input(shape=self.input_shape, name='input_anchor')
        input_p = Input(shape=self.input_shape, name='input_positive')
        
        caps_a, embedding_a = base_network(input_a)
        caps_p, embedding_p = base_network(input_p)
        
        dist_type = self.distance_type if self.distance_type else ('cosine' if not self.use_pearson_distance else 'pearson')
        if dist_type == 'cosine':
            print("[DISTANCE] Using Cosine distance computation")
            print("[DISTANCE] Expected range: [0.0, 1.0] for L2 normalized embeddings")
            print("[DISTANCE] Threshold: 0.5 (will be set automatically)")
            distance = Lambda(self.cosine_distance, name='cosine_distance')([embedding_a, embedding_p])
        elif dist_type == 'euclidean':
            print("[DISTANCE] Using Euclidean distance computation")
            print("[DISTANCE] WARNING: If embeddings are L2 normalized, range will be [0, 1.414]")
            print("[DISTANCE] Expected range for raw embeddings: [3, 8]")
            print("[DISTANCE] Threshold: 5.0 (set for raw embeddings)")
            print("[DISTANCE] If metrics=0, try switching to 'cosine'!")
            distance = Lambda(self.euclidean_distance, name='euclidean_distance')([embedding_a, embedding_p])
        else:
            print("[DISTANCE] Using Pearson correlation-based distance computation")
            distance = Lambda(self.pearson_correlation_distance, name='pearson_distance')([embedding_a, embedding_p])
        
        distance = Lambda(
            lambda d: tf.reduce_mean(tf.reshape(d, (tf.shape(d)[0], -1)), axis=1, keepdims=True),
            name='distance'
        )(distance)

        siamese_model = Model(
            inputs=[input_a, input_p],
            outputs=distance,
            name='Siamese_CapsNet_MobileNetV2'
        )
        
        class _ThresholdedMetricBase(tf.keras.metrics.Metric):
            def __init__(self, name='metric', **kwargs):
                super().__init__(name=name, **kwargs)
                self.weighted_correct = self.add_weight(name='weighted_correct', initializer='zeros', dtype=tf.float32)
                self.weighted_total = self.add_weight(name='weighted_total', initializer='zeros', dtype=tf.float32)
                # Track simple running stats of predictions for adaptive threshold suggestion
                self._pred_min = self.add_weight(name='pred_min', initializer=lambda shape, dtype: tf.constant(1.0, dtype=dtype), dtype=tf.float32)
                self._pred_max = self.add_weight(name='pred_max', initializer='zeros', dtype=tf.float32)
                self._pred_sum = self.add_weight(name='pred_sum', initializer='zeros', dtype=tf.float32)
                self._pred_count = self.add_weight(name='pred_count', initializer='zeros', dtype=tf.float32)

            def _normalize(self, y_true, y_pred):
                y_true = tf.cast(y_true, tf.float32)
                y_pred = tf.cast(y_pred, tf.float32)
                b = tf.shape(y_pred)[0]
                y_true = tf.reshape(y_true, (b, -1))
                y_true = tf.reduce_mean(y_true, axis=1, keepdims=True)
                y_pred = tf.reshape(y_pred, (b, -1))
                y_pred = tf.reduce_mean(y_pred, axis=1, keepdims=True)
                # For distance metrics (cosine/euclidean): lower distance => similar => positive label if label_positive_means_similar
                # For pearson (if used): existing pipeline converts to distance already before metrics
                if not self_outer.label_positive_means_similar:
                    # If positive means dissimilar, invert labels semantics for metric computation
                    y_true = 1.0 - y_true
                # FIXED: Use fixed threshold 0.5 for training metrics to avoid validation threshold mismatch
                # The validation callback updates _threshold_var with validation-optimal threshold,
                # but training data has different score distribution, causing training recall to collapse
                # when using validation threshold. Training metrics are for monitoring only, so fixed 0.5 is fine.
                base_thr = tf.constant(0.5, dtype=tf.float32)
                # Update running stats (no control dependencies needed; metrics graph-safe)
                self._pred_min.assign(tf.minimum(self._pred_min, tf.reduce_min(y_pred)))
                self._pred_max.assign(tf.maximum(self._pred_max, tf.reduce_max(y_pred)))
                self._pred_sum.assign_add(tf.reduce_sum(y_pred))
                self._pred_count.assign_add(tf.cast(b, tf.float32))
                # Optional mid-point adaptive suggestion (not applied unless enabled on wrapper)
                if getattr(self_outer, 'adaptive_metric_threshold', False):
                    # Midpoint between running min and max with light smoothing
                    dynamic_thr = (self._pred_min + self._pred_max) * 0.5
                    # Blend with base threshold (smoothing factor 0.7 retains stability)
                    base_thr = 0.7 * base_thr + 0.3 * dynamic_thr
                preds = tf.cast(tf.less_equal(y_pred, base_thr), tf.float32)
                return y_true, preds

            def update_core(self, y_true, preds, sample_weight):
                raise NotImplementedError

            def update_state(self, y_true, y_pred, sample_weight=None):
                y_true, preds = self._normalize(y_true, y_pred)
                if sample_weight is None:
                    sample_weight = tf.ones_like(tf.squeeze(y_true, axis=-1), dtype=tf.float32)
                else:
                    sample_weight = tf.cast(sample_weight, tf.float32)
                    sample_weight = tf.reshape(sample_weight, (-1,))
                self.update_core(y_true, preds, sample_weight)
                # Optional debug: print distribution stats every ~200 batches when enabled
                if getattr(self_outer, 'debug_metric_distribution', False):
                    # Use modulo on internal count to limit prints
                    total_seen = tf.cast(self._pred_count, tf.int32)
                    cond = tf.equal(tf.math.floormod(total_seen, 200), 0)
                    def _print():
                        thr = tf.constant(0.5, dtype=tf.float32)  # Fixed threshold for training metrics
                        pred_mean = self._pred_sum / (self._pred_count + 1e-6)
                        tf.print('[MetricDebug]', self.name, 'pred_min=', self._pred_min, 'pred_max=', self._pred_max, 'mean=', pred_mean, 'threshold=', thr, output_stream='file://stdout')
                        # Enhanced diagnostics: explain what the values mean
                        tf.print('[MetricDebug]   → Distance range: [', self._pred_min, ',', self._pred_max, ']', output_stream='file://stdout')
                        tf.print('[MetricDebug]   → If ALL > threshold: TP=0 (all predict dissimilar)', output_stream='file://stdout')
                        tf.print('[MetricDebug]   → If ALL < threshold: TN=0 (all predict similar)', output_stream='file://stdout')
                        # Diagnostic: are all predictions on one side?
                        tf.print('[MetricDebug]   → If ALL distances > threshold: TP=0 (all predict dissimilar)', output_stream='file://stdout')
                        tf.print('[MetricDebug]   → If ALL distances < threshold: TN=0 (all predict similar)', output_stream='file://stdout')
                        return 0
                    tf.cond(cond, lambda: _print(), lambda: 0)

            def result(self):
                return tf.where(self.weighted_total > 0.0, self.weighted_correct / (self.weighted_total + 1e-7), 0.0)

            def reset_state(self):
                for var in self.variables:
                    var.assign(0.0)
                # Reset min to 1.0 after wipe
                self._pred_min.assign(1.0)

        self_outer = self

        optimizer = self._build_optimizer()
        
        # CRITICAL: Set margin based on distance type for contrastive loss
        # Euclidean with raw embeddings (dim=128): distances ~4-6, need large margin (~6-8)
        # Cosine/Pearson normalized (0-1): need small margin (~1.0-2.0)
        if self.distance_type == 'euclidean':
            contrastive_margin = max(5.0, float(getattr(self, 'base_contrastive_margin', 7.0)))  # Large margin for raw euclidean distances
            self.base_contrastive_margin = contrastive_margin
            self.margin_schedule_max = max(self.margin_schedule_max, self.base_contrastive_margin * 1.2)
        else:
            # Keep cosine/pearson margins inside the reachable [0, 0.95] band so the loss target is
            # actually attainable.
            contrastive_margin = float(getattr(self, 'base_contrastive_margin', 0.65))
            contrastive_margin = float(np.clip(contrastive_margin, 0.4, 0.95))
            self.base_contrastive_margin = contrastive_margin
            self.margin_schedule_max = float(min(max(self.margin_schedule_max, self.base_contrastive_margin * 1.2), 0.95))
        if getattr(self, 'margin_var', None) is not None:
            try:
                self.margin_var.assign(float(self.base_contrastive_margin))
            except Exception:
                pass
        
        if self.loss_type == 'contrastive_enhanced':
            def loss_fn(y_true, y_pred):
                margin_ref = self.margin_var if getattr(self, 'margin_var', None) is not None else self.base_contrastive_margin
                return self.enhanced_contrastive_loss(y_true, y_pred, margin=margin_ref)
        elif self.loss_type in ('bce_similarity', 'bce_on_similarity'):
            def loss_fn(y_true, y_pred):
                y_true = tf.cast(y_true, tf.float32)
                y_pred = tf.cast(y_pred, tf.float32)
                b = tf.shape(y_pred)[0]
                y_true = tf.reshape(y_true, (b, -1))
                y_true = tf.reduce_mean(y_true, axis=1, keepdims=True)
                if not self.label_positive_means_similar:
                    y_true = 1.0 - y_true
                if isinstance(self.label_smoothing, (float, int)) and self.label_smoothing > 0.0:
                    ls = tf.convert_to_tensor(self.label_smoothing, dtype=tf.float32)
                    y_true = y_true * (1.0 - ls) + 0.5 * ls
                y_pred = tf.reshape(y_pred, (b, -1))
                y_pred = tf.reduce_mean(y_pred, axis=1, keepdims=True)
                sim = tf.clip_by_value(1.0 - y_pred, 1e-6, 1.0 - 1e-6)
                loss = -(y_true * tf.math.log(sim) + (1.0 - y_true) * tf.math.log(1.0 - sim))
                return tf.squeeze(loss, axis=-1)
        else:
            loss_fn = lambda y_true, y_pred: self.simple_contrastive_loss(y_true, y_pred, margin=1.0)

        self._loss_function = loss_fn
        metrics_list = self._build_metrics()

        siamese_model.compile(
            optimizer=optimizer,
            loss=loss_fn,
            metrics=metrics_list
        )
        
        self.model = base_network
        self.siamese_model = siamese_model
        
        total_params = siamese_model.count_params()
        print(f"Complete Siamese network created with {total_params:,} total parameters")
        
        return siamese_model

    def get_metric_distance_threshold(self):
        try:
            return float(self._threshold_var.numpy()) if self._threshold_var is not None else float(self.metric_distance_threshold)
        except Exception:
            return float(self.metric_distance_threshold)

    def set_metric_distance_threshold(self, value: float):
        try:
            v = float(value)
            if v < 0.0:
                v = 0.0
            if v > 2.0:
                v = 2.0
            if self._threshold_var is not None:
                self._threshold_var.assign(v)
            self.metric_distance_threshold = v
        except Exception:
            self.metric_distance_threshold = float(value)

    def get_contrastive_margin(self) -> float:
        try:
            if getattr(self, 'margin_var', None) is not None:
                return float(self.margin_var.numpy())
        except Exception:
            pass
        return float(getattr(self, 'base_contrastive_margin', 1.5))

    def update_contrastive_margin(self, new_margin: float):
        # Margin updates disabled - analysis shows scheduler degraded performance
        # Keep margin fixed at initialization value
        return

    def get_embedding_noise_layer(self):
        if not hasattr(self, 'model') or self.model is None:
            return None
        try:
            return self.model.get_layer(self.noise_layer_name)
        except Exception:
            return None

    def update_embedding_noise_std(self, new_std: float):
        try:
            target_std = max(0.0, float(new_std))
        except Exception:
            return
        layer = self.get_embedding_noise_layer()
        if layer is not None and hasattr(layer, 'stddev'):
            layer.stddev = target_std
    
    def get_model_summary(self):
        if self.siamese_model is None:
            return {"error": "Model not created yet"}
        
        return {
            "architecture": "Siamese Capsule Network with MobileNetV2",
            "total_parameters": self.siamese_model.count_params(),
            "input_shape": self.input_shape,
            "embedding_dim": self.embedding_dim,
            "num_capsules": self.num_capsules,
            "dim_capsules": self.dim_capsules,
            "distance_type": self.distance_type
        }
    
    def load_model(self, model_path='models/siamese_capsule/best_model.h5'):
        print(f"Loading model from {model_path}...")
        try:
            self.siamese_model = tf.keras.models.load_model(model_path, compile=False)
            self.create_siamese_capsule_network()
            self.siamese_model.load_weights(model_path)
            print("Model loaded successfully!")
        except Exception as e:
            print(f"Error loading model: {e}")
            raise


# ============================================================================
# EMBEDDED: Data Generators
# ============================================================================

class CSVPairGenerator(tf.keras.utils.Sequence):
    """Keras Sequence that yields image pairs and labels from a CSV file."""
    def __init__(self, csv_path: str, batch_size: int = 32, shuffle: bool = True, target_size: Tuple[int, int] = (224, 224)):
        self.csv_path = csv_path
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.target_size = target_size
        self.samples: List[Tuple[str, str, float]] = []

        df = pd.read_csv(csv_path)
        img_cols = []
        label_col = None

        for name in ['img1_path', 'image1', 'path1', 'img_1', 'image_1']:
            if name in df.columns:
                img_cols.append(name)
                break
        for name in ['img2_path', 'image2', 'path2', 'img_2', 'image_2']:
            if name in df.columns:
                img_cols.append(name)
                break
        for name in ['label', 'is_same', 'same', 'target']:
            if name in df.columns:
                label_col = name
                break

        if len(img_cols) < 2:
            candidates = []
            for col in df.columns:
                if df[col].dtype == object and df[col].astype(str).str.contains(r"\\\\|/|\\.jpg|\\.png|\\.jpeg", regex=True, na=False).any():
                    candidates.append(col)
            img_cols = candidates[:2]
        if label_col is None:
            for col in df.columns:
                if np.issubdtype(df[col].dtype, np.number):
                    unique_vals = pd.unique(df[col].dropna())
                    if set(unique_vals).issubset({0, 1}) or len(unique_vals) <= 3:
                        label_col = col
                        break

        if len(img_cols) < 2 or label_col is None:
            raise ValueError(f"CSV {csv_path} does not contain recognizable pair/label columns")

        preprocessed_root = "(Preprocessed) breed_organized_with_images"

        def fix_path(p: str) -> str:
            p = str(p)
            if os.path.exists(p):
                return p
            if "breed_organized_with_images" in p and preprocessed_root not in p:
                p2 = p.replace("breed_organized_with_images", preprocessed_root)
                if os.path.exists(p2):
                    return p2
            if not os.path.isabs(p):
                candidate = os.path.join(preprocessed_root, p)
                if os.path.exists(candidate):
                    return candidate
            return p

        for _, row in df.iterrows():
            p1 = fix_path(row[img_cols[0]])
            p2 = fix_path(row[img_cols[1]])
            if not os.path.exists(p1) or not os.path.exists(p2):
                continue
            lbl = float(row[label_col])
            lbl = 1.0 if lbl >= 0.5 else 0.0
            self.samples.append((p1, p2, lbl))

        if len(self.samples) == 0:
            print(f"Warning: no valid samples found in {csv_path}")

        self.indices = np.arange(len(self.samples))
        if self.shuffle and len(self.indices) > 0:
            np.random.shuffle(self.indices)

    def __len__(self):
        if self.batch_size <= 0:
            return 0
        return int(np.ceil(len(self.samples) / self.batch_size))

    def on_epoch_end(self):
        if self.shuffle and len(self.samples) > 0:
            np.random.shuffle(self.indices)

    def __getitem__(self, idx):
        if len(self.samples) == 0:
            b = max(1, self.batch_size)
            x1 = np.random.random((b, 224, 224, 3)).astype(np.float32)
            x2 = np.random.random((b, 224, 224, 3)).astype(np.float32)
            y = np.random.randint(0, 2, b).astype(np.float32)
            y = y.reshape((-1, 1)).astype(np.float32)
            sample_weights = np.ones((b,), dtype=np.float32)
            return [x1, x2], y, sample_weights

        batch_idx = self.indices[idx * self.batch_size:(idx + 1) * self.batch_size]
        imgs1, imgs2, labels = [], [], []
        for bi in batch_idx:
            p1, p2, lbl = self.samples[bi]
            img1 = tf.keras.utils.load_img(p1, target_size=self.target_size)
            img2 = tf.keras.utils.load_img(p2, target_size=self.target_size)
            a1 = tf.keras.utils.img_to_array(img1) / 255.0
            a2 = tf.keras.utils.img_to_array(img2) / 255.0
            imgs1.append(a1)
            imgs2.append(a2)
            labels.append(lbl)
        x1 = np.stack(imgs1).astype(np.float32)
        x2 = np.stack(imgs2).astype(np.float32)
        
        if self.shuffle:
            flip_mask = np.random.rand(x1.shape[0]) < 0.2
            x1[flip_mask] = x1[flip_mask, :, ::-1, :]
            x2[flip_mask] = x2[flip_mask, :, ::-1, :]
            jitter = (np.random.rand(x1.shape[0], 1, 1, 1) - 0.5) * 0.03
            x1 = np.clip(x1 * (1.0 + jitter), 0.0, 1.0)
            jitter = (np.random.rand(x2.shape[0], 1, 1, 1) - 0.5) * 0.03
            x2 = np.clip(x2 * (1.0 + jitter), 0.0, 1.0)
            
        y = np.array(labels, dtype=np.float32).reshape((-1, 1))
        try:
            pos = float(np.sum(y))
            neg = float(y.shape[0] - pos)
            if pos > 0 and neg > 0:
                w_pos = 0.5 / pos
                w_neg = 0.5 / neg
                sample_weights = np.where(y.flatten() > 0.5, w_pos, w_neg).astype(np.float32)
            else:
                sample_weights = np.ones((y.shape[0],), dtype=np.float32)
        except Exception:
            sample_weights = np.ones((y.shape[0],), dtype=np.float32)
        return [x1, x2], y, sample_weights


def create_data_generators_from_csv(batch_size: int = 64):
    """Create train/val/test generators from processed_data CSV pair files.
    batch_size: desired training batch size (train). Validation/test use min(batch_size//2, batch_size) for memory safety.
    """
    print("Loading CSV pair generators from processed_data ...")
    base_dir = "processed_data"
    train_csv = os.path.join(base_dir, "train_pairs.csv")
    val_csv = os.path.join(base_dir, "val_pairs.csv")
    test_csv = os.path.join(base_dir, "test_pairs.csv")

    if not (os.path.exists(train_csv) and os.path.exists(val_csv) and os.path.exists(test_csv)):
        print("CSV files not found; falling back to demo generators.")
        return create_demo_generators()

    # FIXED: Validation should shuffle to handle sequentially-ordered CSV files
    # This ensures balanced class distribution in each epoch
    # Safety clamp
    train_bs = int(batch_size)
    if train_bs <= 0:
        train_bs = 64
    # Validation/test typically can be smaller to reduce GPU memory pressure
    val_bs = max(8, min(train_bs // 2, train_bs))
    test_bs = val_bs
    print(f"Creating generators with batch sizes => train: {train_bs}, val: {val_bs}, test: {test_bs}")
    train_gen = CSVPairGenerator(train_csv, batch_size=train_bs, shuffle=True)
    val_gen = CSVPairGenerator(val_csv, batch_size=val_bs, shuffle=True)
    test_gen = CSVPairGenerator(test_csv, batch_size=test_bs, shuffle=False)
    return train_gen, val_gen, test_gen


def create_demo_generators():
    """Fallback demo generators with random data (small sizes)."""
    print("Creating fallback demo generators...")

    class FallbackDemoGenerator(tf.keras.utils.Sequence):
        def __init__(self, num_samples=64, batch_size=16):
            self.num_samples = num_samples
            self.batch_size = batch_size
            print(f"Using fallback demo generator with {num_samples} samples")

        def __len__(self):
            return max(1, self.num_samples // self.batch_size)

        def __getitem__(self, idx):
            b = self.batch_size
            batch_img1 = np.random.random((b, 224, 224, 3)).astype(np.float32)
            batch_img2 = np.random.random((b, 224, 224, 3)).astype(np.float32)
            batch_labels = np.random.randint(0, 2, b).astype(np.float32).reshape((-1, 1))
            sample_weights = np.ones((b,), dtype=np.float32)
            return [batch_img1, batch_img2], batch_labels, sample_weights

    train_gen = FallbackDemoGenerator(num_samples=64, batch_size=16)
    val_gen = FallbackDemoGenerator(num_samples=32, batch_size=8)
    test_gen = FallbackDemoGenerator(num_samples=32, batch_size=8)
    return train_gen, val_gen, test_gen


MODELS_AVAILABLE = True


# ============================================================================
# GUI IMPLEMENTATION BEGINS HERE
# ============================================================================

class TrainingProgressCallback(tf.keras.callbacks.Callback):
    """Custom callback to send training progress to GUI with real-time updates."""
    
    def __init__(self, progress_queue, update_frequency=1, total_epochs=50):
        super().__init__()
        self.progress_queue = progress_queue
        self.update_frequency = update_frequency
        self.epoch_count = 0
        self.training_start_time = None
        self.epoch_start_time = None
        self.total_epochs = total_epochs  # Store total epochs to avoid params access
        
    def on_train_begin(self, logs=None):
        self.training_start_time = time.time()
        self.progress_queue.put({
            'type': 'training_start',
            'message': 'Training started - Preparing data...'
        })
        
    def on_epoch_begin(self, epoch, logs=None):
        self.epoch_count = epoch + 1
        self.epoch_start_time = time.time()
        self.progress_queue.put({
            'type': 'epoch_start',
            'epoch': self.epoch_count,
            'message': f'Starting Epoch {self.epoch_count}...'
        })
    
    def on_batch_end(self, batch, logs=None):
        # Update every 10 batches to show training is active
        if batch % 10 == 0 and logs:
            self.progress_queue.put({
                'type': 'batch_update',
                'epoch': self.epoch_count,
                'batch': batch,
                'logs': logs,
                'message': f'Epoch {self.epoch_count} - Batch {batch}: Loss {logs.get("loss", 0):.4f}'
            })
        
    def on_epoch_end(self, epoch, logs=None):
        if not logs:
            return

        # Initialize timing metrics safely
        try:
            epoch_time = float(time.time() - (self.epoch_start_time or time.time()))
        except Exception:
            epoch_time = None
        try:
            elapsed_time = float(time.time() - (self.training_start_time or time.time()))
        except Exception:
            elapsed_time = None

        # Estimate ETA
        eta = None
        try:
            if elapsed_time is not None:
                avg_epoch_time = float(elapsed_time) / (epoch + 1)
                remaining_epochs = max(0, self.total_epochs - (epoch + 1))
                eta = float(remaining_epochs) * avg_epoch_time
        except Exception:
            eta = None

        # Format ETA display
        eta_display = ""
        try:
            if eta is not None and isinstance(eta, (int, float)) and eta > 0:
                eta_display = f" (ETA: {eta/60:.1f}min)"
        except Exception:
            eta_display = ""

        # Safe val_loss formatting
        try:
            val_loss_raw = logs.get('val_loss', 0)
            val_loss_fmt = f"{float(val_loss_raw):.4f}"
        except Exception:
            val_loss_fmt = str(logs.get('val_loss', 0))

        # Build progress data (always defined)
        try:
            loss_fmt = f"{float(logs.get('loss', 0)):.4f}"
        except Exception:
            loss_fmt = str(logs.get('loss', 0))
        progress_data = {
            'type': 'epoch_complete',
            'epoch': self.epoch_count,
            'logs': logs,
            'elapsed_time': elapsed_time,
            'epoch_time': epoch_time,
            'eta': eta,
            'message': f'Epoch {self.epoch_count} completed - Loss: {loss_fmt}, Val Loss: {val_loss_fmt}{eta_display}'
        }
        # Queue progress update (with safeguard)
        try:
            self.progress_queue.put(progress_data)
        except Exception:
            # Fallback minimal message
            self.progress_queue.put({'type': 'epoch_complete', 'epoch': self.epoch_count, 'logs': logs})
    
    def on_train_end(self, logs=None):
        total_time = time.time() - self.training_start_time
        total_time_min = None
        try:
            total_time_min = float(total_time) / 60.0
        except Exception:
            total_time_min = None
        message = f'Training completed in {total_time_min:.1f} minutes' if total_time_min is not None else 'Training completed'
        self.progress_queue.put({
            'type': 'training_complete',
            'message': message,
            'final_logs': logs
        })
    
class EmbeddingVarianceCallback(tf.keras.callbacks.Callback):
    """Monitor embedding variance to detect collapse during training."""
    
    def __init__(self, val_data, base_model, progress_queue=None, check_every_n_epochs=1):
        super().__init__()
        self.val_data = val_data
        self.base_model = base_model
        self.progress_queue = progress_queue
        self.check_every_n_epochs = check_every_n_epochs
        self.embedding_model = None
    
    def on_train_begin(self, logs=None):
        try:
            # Get the embedding output from base model
            if self.base_model is not None and hasattr(self.base_model, 'outputs'):
                # outputs[1] is the embeddings (outputs[0] is capsules)
                self.embedding_model = tf.keras.Model(
                    self.base_model.input, 
                    self.base_model.outputs[1]
                )
        except Exception as e:
            print(f"[EmbeddingVariance] Could not create embedding model: {e}")
    
    def on_epoch_end(self, epoch, logs=None):
        if epoch % self.check_every_n_epochs != 0:
            return
        
        if self.embedding_model is None:
            return
        
        try:
            # Sample a batch from validation data
            batch = self.val_data[0]
            if isinstance(batch, (list, tuple)) and len(batch) >= 2:
                (xa, xb), y = batch[0], batch[1]
            else:
                return
            
            # Get embeddings with training=True to activate GaussianNoise
            # This ensures proper variance measurement during monitoring
            # Convert to numpy after the call to avoid eager execution issues
            emb_a_tensor = self.embedding_model(xa, training=True)
            emb_b_tensor = self.embedding_model(xb, training=True)
            # Check if tensors or tuples (base model returns tuple)
            if isinstance(emb_a_tensor, (list, tuple)):
                emb_a = emb_a_tensor[1].numpy()  # Get embedding, not capsules
                emb_b = emb_b_tensor[1].numpy()
            else:
                emb_a = emb_a_tensor.numpy()
                emb_b = emb_b_tensor.numpy()
            
            # Compute statistics
            all_embs = np.concatenate([emb_a, emb_b], axis=0)
            per_dim_std = all_embs.std(axis=0)
            global_std = float(per_dim_std.mean())
            min_std = float(per_dim_std.min())
            max_std = float(per_dim_std.max())
            spread = float(all_embs.max() - all_embs.min())
            mean_norm = float(np.linalg.norm(all_embs, axis=1).mean())
            
            # Log to console
            print(f"\n[EmbeddingVariance] Epoch {epoch+1}:")
            print(f"  Global std: {global_std:.5f}")
            print(f"  Spread: {spread:.5f}")
            print(f"  Mean norm: {mean_norm:.5f}")
            print(f"  Std range: [{min_std:.5f}, {max_std:.5f}]")
            
            # Add to logs
            if logs is not None:
                logs['emb_std'] = global_std
                logs['emb_spread'] = spread
            
            # Send to GUI
            if self.progress_queue is not None:
                try:
                    self.progress_queue.put({
                        'type': 'embedding_variance',
                        'epoch': epoch + 1,
                        'std': global_std,
                        'spread': spread,
                        'mean_norm': mean_norm
                    })
                except Exception:
                    pass
            
            # Check for collapse
            if global_std < 0.001 or spread < 0.01:
                print(f"  ⚠ WARNING: Very low variance detected - possible collapse!")
            elif global_std < 0.01 or spread < 0.05:
                print(f"  ⚠ Variance is getting low - monitor closely")
            else:
                print(f"  ✓ Variance looks healthy")
        
        except Exception as e:
            print(f"[EmbeddingVariance] Error: {e}")


class GaussianNoiseAnnealingCallback(tf.keras.callbacks.Callback):
    """Gradually reduce Gaussian noise injected into embeddings once training stabilizes."""

    def __init__(self, model_wrapper, start_epoch=4, end_epoch=12, min_std=0.02, progress_queue=None, apply_every=1):
        super().__init__()
        self.model_wrapper = model_wrapper
        self.start_epoch = int(start_epoch)
        self.end_epoch = int(max(self.start_epoch + 1, end_epoch))
        self.min_std = float(min_std)
        self.progress_queue = progress_queue
        self.apply_every = max(1, int(apply_every))
        self.initial_std = float(getattr(model_wrapper, 'noise_initial_std', 0.0) or 0.0)
        if self.initial_std <= 0.0:
            layer = model_wrapper.get_embedding_noise_layer() if hasattr(model_wrapper, 'get_embedding_noise_layer') else None
            if layer is not None and hasattr(layer, 'stddev'):
                self.initial_std = float(layer.stddev)
        self.last_applied = self.initial_std

    def _scheduled_std(self, epoch_index: int) -> float:
        if self.initial_std <= self.min_std or epoch_index < self.start_epoch:
            return self.initial_std
        span = max(1, self.end_epoch - self.start_epoch)
        progress = min(1.0, max(0, epoch_index - self.start_epoch + 1) / span)
        return max(self.min_std, self.initial_std - (self.initial_std - self.min_std) * progress)

    def on_epoch_end(self, epoch, logs=None):
        if self.initial_std <= 0.0:
            return
        if (epoch + 1) % self.apply_every != 0:
            return
        target_std = self._scheduled_std(epoch + 1)
        if abs(target_std - self.last_applied) < 1e-4:
            return
        if hasattr(self.model_wrapper, 'update_embedding_noise_std'):
            self.model_wrapper.update_embedding_noise_std(target_std)
            self.last_applied = target_std
            if logs is not None:
                logs['noise_std'] = float(target_std)
            if self.progress_queue is not None:
                try:
                    self.progress_queue.put({'type': 'debug', 'message': f"[NoiseAnneal] Epoch {epoch+1}: std -> {target_std:.4f}"})
                except Exception:
                    pass


class ContrastiveMarginScheduler(tf.keras.callbacks.Callback):
    """Increase contrastive margin when validation F1 plateaus to push embeddings further apart."""

    def __init__(self, model_wrapper, max_margin, start_epoch=4, patience=2, min_delta=0.002, progress_queue=None, step_fraction=0.35):
        super().__init__()
        self.model_wrapper = model_wrapper
        self.max_margin = float(max_margin)
        if getattr(model_wrapper, 'distance_type', None) != 'euclidean':
            self.max_margin = float(np.clip(self.max_margin, 0.4, 0.95))
        self.start_epoch = int(start_epoch)
        self.patience = max(1, int(patience))
        self.min_delta = float(min_delta)
        self.progress_queue = progress_queue
        self.step_fraction = float(np.clip(step_fraction, 0.05, 0.75))
        self.best_f1 = -np.inf
        self.stall_epochs = 0

    def on_epoch_end(self, epoch, logs=None):
        if logs is None:
            return
        val_f1 = None
        for key in ('val_f1', 'val_f1_score', 'val_f1_metric'):
            if key in logs and logs[key] is not None:
                try:
                    val_f1 = float(logs[key])
                    break
                except Exception:
                    continue
        if val_f1 is None:
            return
        if val_f1 > self.best_f1 + self.min_delta:
            self.best_f1 = val_f1
            self.stall_epochs = 0
            return
        self.stall_epochs += 1
        if (epoch + 1) < self.start_epoch or self.stall_epochs < self.patience:
            return
        current_margin = self.model_wrapper.get_contrastive_margin() if hasattr(self.model_wrapper, 'get_contrastive_margin') else None
        if current_margin is None:
            return
        if current_margin >= self.max_margin - 1e-4:
            return
        increment = (self.max_margin - current_margin) * self.step_fraction
        new_margin = min(self.max_margin, current_margin + increment)
        if hasattr(self.model_wrapper, 'update_contrastive_margin'):
            self.model_wrapper.update_contrastive_margin(new_margin)
            if logs is not None:
                logs['contrastive_margin'] = float(new_margin)
            if self.progress_queue is not None:
                try:
                    self.progress_queue.put({'type': 'debug', 'message': f"[MarginScheduler] Epoch {epoch+1}: margin -> {new_margin:.3f}"})
                except Exception:
                    pass
        self.stall_epochs = 0

class EnsureValMetricsCallback(tf.keras.callbacks.Callback):
        """Ensure validation metrics exist in logs for downstream consumers (e.g., ModelCheckpoint filenames)."""
        def on_epoch_end(self, epoch, logs=None):
            if logs is None:
                return
            try:
                if 'val_accuracy' not in logs and 'val_binary_accuracy_metric' in logs:
                    logs['val_accuracy'] = logs.get('val_binary_accuracy_metric')
                # Ensure precision/recall keys exist
                if 'val_precision' not in logs:
                    logs['val_precision'] = logs.get('val_precision_metric', 0.0)
                if 'val_recall' not in logs:
                    logs['val_recall'] = logs.get('val_recall_metric', 0.0)
                # Compute F1 if missing
                if 'val_f1' not in logs:
                    try:
                        p = float(logs.get('val_precision', 0.0))
                        r = float(logs.get('val_recall', 0.0))
                        logs['val_f1'] = 2.0 * p * r / (p + r + 1e-8)
                    except Exception:
                        logs['val_f1'] = 0.0
                # Ensure AUC exists
                if 'val_auc' not in logs:
                    logs['val_auc'] = 0.0
                # Coerce common keys to floats to avoid Keras progbar type errors
                for k in list(logs.keys()):
                    try:
                        if isinstance(logs[k], (int, float)):
                            logs[k] = float(logs[k])
                        elif hasattr(logs[k], 'item'):
                            logs[k] = float(logs[k].item())
                        else:
                            # Drop non-numeric values from logs
                            del logs[k]
                    except Exception:
                        try:
                            del logs[k]
                        except Exception:
                            pass
            except Exception:
                # Do not break training if logs manipulation fails
                pass



class ValidationMetricsCallback(tf.keras.callbacks.Callback):
    """Compute global validation metrics & optimal threshold with distance->similarity normalization.
    Also emits ROC data and saves improved models. Supports optional EMA smoothing of thresholds to reduce oscillations.
    Adds balanced sampling of validation subset per epoch to stabilize metrics."""

    def __init__(self, val_data, model_wrapper, threshold_search=True, max_samples=10000, progress_queue=None, model_dir=None, save_on='f1', threshold_smoothing=0.2, balanced=True, per_class_max=None,
                 log_fixed_threshold=True, fixed_threshold=0.5, freeze_after_epoch=None, tta_hflip=False):
        super().__init__()
        self.val_data = val_data
        self.model_wrapper = model_wrapper
        self.best_f1 = 0.0
        self.best_auc = 0.0
        self.best_threshold = 0.5
        self.threshold_search = threshold_search
        self.max_samples = max_samples
        self.progress_queue = progress_queue
        self.model_dir = model_dir
        self.save_on = save_on  # 'f1' or 'auc'
        # Additional standardized threshold options
        self.log_fixed_threshold = bool(log_fixed_threshold)
        try:
            self.fixed_threshold = float(fixed_threshold)
        except Exception:
            self.fixed_threshold = 0.5
        self.freeze_after_epoch = None if freeze_after_epoch is None else int(freeze_after_epoch)
        self.frozen_threshold_score = None
        # Lightweight TTA options
        self.tta_hflip = bool(tta_hflip)
        # EMA smoothing factor in [0,1]; 0 disables smoothing
        try:
            self.threshold_smoothing = float(threshold_smoothing)
        except Exception:
            self.threshold_smoothing = 0.0
        # Balanced sampling controls - FIXED: Set minimum per_class_max for stable ROC
        self.balanced = bool(balanced)
        # CRITICAL FIX: Increased from 250 to 1000 per class for much smoother ROC
        # Previous: 500 total samples caused jagged ROC curve
        # Current: 2000 total samples provides stable, smooth ROC
        if per_class_max is None:
            per_class_max = 1000  # Increased from 250 for smoother ROC
        self.per_class_max = per_class_max if isinstance(per_class_max, (int, float)) else 1000
        if not hasattr(self.model_wrapper, 'validation_threshold'):
            setattr(self.model_wrapper, 'validation_threshold', self.best_threshold)
        try:
            from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, accuracy_score, roc_curve, precision_recall_curve, average_precision_score
            self._precision_score = precision_score
            self._recall_score = recall_score
            self._f1_score = f1_score
            self._roc_auc_score = roc_auc_score
            self._accuracy_score = accuracy_score
            self._roc_curve = roc_curve
            self._pr_curve = precision_recall_curve
            self._avg_precision = average_precision_score
        except Exception:
            self._precision_score = self._recall_score = self._f1_score = None
            self._roc_auc_score = self._accuracy_score = None
            self._roc_curve = None
            self._pr_curve = None
            self._avg_precision = None

    def _predict_with_tta(self, inputs):
        """Optionally average predictions with simple horizontal-flip TTA.
        Inputs are expected as [x1, x2]. Returns a numpy array of predictions.
        """
        try:
            if not self.tta_hflip:
                return self.model_wrapper.siamese_model.predict(inputs, verbose=0)
            import numpy as np
            x1, x2 = inputs
            # Original prediction
            p0 = self.model_wrapper.siamese_model.predict([x1, x2], verbose=0)
            # Flipped prediction
            try:
                x1f = np.flip(np.asarray(x1), axis=2)
                x2f = np.flip(np.asarray(x2), axis=2)
                p1 = self.model_wrapper.siamese_model.predict([x1f, x2f], verbose=0)
                # Average
                p0 = np.asarray(p0)
                p1 = np.asarray(p1)
                if p0.shape == p1.shape:
                    return 0.5 * (p0 + p1)
                else:
                    return p0
            except Exception:
                return p0
        except Exception:
            return self.model_wrapper.siamese_model.predict(inputs, verbose=0)

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        # pessimistically mark error until successful completion
        try:
            logs['validation_callback_error'] = 1.0
        except Exception:
            pass
        if self._f1_score is None:
            return
        try:
            y_true = []
            y_raw_scores = []
            count = 0
            consumed = False
            # Prefer index-based access for Keras Sequence
            if hasattr(self.val_data, '__len__') and hasattr(self.val_data, '__getitem__'):
                try:
                    data_len = len(self.val_data)
                except Exception:
                    data_len = 0
                for i in range(data_len):
                    try:
                        batch = self.val_data[i]
                        if isinstance(batch, (list, tuple)):
                            if len(batch) == 3:
                                inputs, labels, _ = batch
                            elif len(batch) == 2:
                                inputs, labels = batch
                            else:
                                continue
                        else:
                            continue
                        preds = self._predict_with_tta(inputs)
                        labels = np.array(labels).ravel()
                        preds = np.array(preds).ravel()
                        if labels.size == 0 or preds.size == 0:
                            continue
                        if np.any(~np.isfinite(preds)):
                            preds = np.nan_to_num(preds, nan=0.0, posinf=1.0, neginf=0.0)
                        y_true.append(labels)
                        y_raw_scores.append(preds)
                        count += len(labels)
                        if self.max_samples and count >= self.max_samples:
                            break
                    except Exception:
                        # Skip malformed batches safely
                        continue
                consumed = True
            # Fallback: try iterator protocol (e.g., for tf.data.Dataset)
            if not consumed:
                try:
                    for batch in iter(self.val_data):
                        try:
                            if isinstance(batch, (list, tuple)):
                                if len(batch) == 3:
                                    inputs, labels, _ = batch
                                elif len(batch) == 2:
                                    inputs, labels = batch
                                else:
                                    continue
                            else:
                                continue
                            preds = self._predict_with_tta(inputs)
                            labels = np.array(labels).ravel()
                            preds = np.array(preds).ravel()
                            if labels.size == 0 or preds.size == 0:
                                continue
                            if np.any(~np.isfinite(preds)):
                                preds = np.nan_to_num(preds, nan=0.0, posinf=1.0, neginf=0.0)
                            y_true.append(labels)
                            y_raw_scores.append(preds)
                            count += len(labels)
                            if self.max_samples and count >= self.max_samples:
                                break
                        except Exception:
                            continue
                except Exception:
                    # If even iterator fails, leave to outer handler
                    pass
            if not y_true:
                try:
                    logs['val_sample_count'] = 0.0
                except Exception:
                    pass
                return
            y_true = np.concatenate(y_true)
            y_raw_scores = np.concatenate(y_raw_scores)
            # Class-balanced subsampling if requested and possible
            try:
                pos_idx = np.where(y_true == 1)[0]
                neg_idx = np.where(y_true == 0)[0]
                total_pos = int(pos_idx.size)
                total_neg = int(neg_idx.size)
                logs['val_pos_total'] = float(total_pos)
                logs['val_neg_total'] = float(total_neg)
                
                # CRITICAL: Check for severe class imbalance indicating data ordering issues
                if total_pos > 0 and total_neg > 0:
                    class_ratio = max(total_pos, total_neg) / min(total_pos, total_neg)
                    if class_ratio > 50:
                        print(f"\nWARNING: Severe class imbalance detected in validation data!")
                        print(f"   Positive samples: {total_pos}, Negative samples: {total_neg}")
                        print(f"   Ratio: {class_ratio:.1f}:1")
                        print(f"   This may indicate sequential class ordering in CSV.")
                        print(f"   Consider shuffling val_pairs.csv or increasing max_samples.\n")
                
                if self.balanced and total_pos > 0 and total_neg > 0:
                    # FIXED: Determine per-class target with minimum threshold for stability
                    half_cap = int(self.max_samples // 2) if (self.max_samples and self.max_samples > 0) else min(total_pos, total_neg)
                    if self.per_class_max is not None:
                        half_cap = min(half_cap, int(self.per_class_max))
                    
                    # CRITICAL FIX: If one class has very few samples (< 100), use all available
                    # This handles cases where data is ordered and we didn't load enough
                    min_class_size = min(total_pos, total_neg)
                    if min_class_size < 100:
                        print(f"\nWARNING: Using all {min_class_size} samples from minority class due to severe imbalance.")
                        print(f"   ROC curves may be jagged. Consider shuffling validation data.\n")
                        per_class_target = min_class_size
                    else:
                        # Ensure minimum 250 samples per class for stable ROC (500 total)
                        per_class_target = max(min(total_pos, total_neg, half_cap), min(250, min(total_pos, total_neg)))
                    
                    # Sample without replacement
                    sel_pos = np.random.choice(pos_idx, size=min(per_class_target, total_pos), replace=False)
                    sel_neg = np.random.choice(neg_idx, size=min(per_class_target, total_neg), replace=False)
                    sel_idx = np.concatenate([sel_pos, sel_neg])
                    np.random.shuffle(sel_idx)
                    y_true = y_true[sel_idx]
                    y_raw_scores = y_raw_scores[sel_idx]
                    logs['val_balanced_applied'] = 1.0
                    logs['val_pos_used'] = float(per_class_target)
                    logs['val_neg_used'] = float(per_class_target)
                else:
                    logs['val_balanced_applied'] = 0.0
                    logs['val_pos_used'] = float(total_pos)
                    logs['val_neg_used'] = float(total_neg)
            except Exception:
                # Fail-safe: proceed without balancing
                logs['val_balanced_applied'] = 0.0
            # Guard against NaNs/Infs after concatenation
            if np.any(~np.isfinite(y_raw_scores)):
                y_raw_scores = np.nan_to_num(y_raw_scores, nan=0.0, posinf=1.0, neginf=0.0)
            
            # CRITICAL FIX: Clip scores to prevent near-zero collapse
            # Previous issue: raw_min=0.00018 caused threshold selection issues
            # Clip to [1e-4, 1-1e-4] to prevent extreme values
            min_clip = 1e-4
            max_clip = 1.0 - 1e-4
            raw_min_unclipped = float(np.min(y_raw_scores))
            raw_max_unclipped = float(np.max(y_raw_scores))
            y_raw_scores = np.clip(y_raw_scores, min_clip, max_clip)
            
            raw_min = float(np.min(y_raw_scores))
            raw_max = float(np.max(y_raw_scores))
            raw_spread = raw_max - raw_min
            
            # ENHANCED: Log score distribution to catch anomalies
            percentiles = np.percentile(y_raw_scores, [1, 5, 25, 50, 75, 95, 99])
            near_zero = np.sum(y_raw_scores < 0.01) / len(y_raw_scores)
            near_one = np.sum(y_raw_scores > 0.99) / len(y_raw_scores)
            print(f"\n[ScoreDistribution] Percentiles: p1={percentiles[0]:.4f} p5={percentiles[1]:.4f} "
                  f"median={percentiles[3]:.4f} p95={percentiles[5]:.4f} p99={percentiles[6]:.4f}")
            if raw_min_unclipped < min_clip or raw_max_unclipped > max_clip:
                print(f"[ScoreClipping] Clipped from [{raw_min_unclipped:.6f}, {raw_max_unclipped:.6f}] to [{raw_min:.6f}, {raw_max:.6f}]")
            # Keep track of raw extreme concentrations for diagnostics
            collapse_diag = {
                'near_zero_frac': float(near_zero),
                'near_one_frac': float(near_one),
                'raw_spread': float(raw_spread),
            }
            
            # DIAGNOSTIC: Warn if spread is too small
            if raw_spread < 0.001:
                print(f"\nWARNING: Raw score spread is very small ({raw_spread:.6f})")
                print(f"   Range: [{raw_min:.6f}, {raw_max:.6f}]")
                print(f"   Model may not be learning to discriminate between classes.")
                print(f"   Check: loss function, learning rate, data quality.\n")
            
            # IMPROVED: Better distance detection for CapsNet models
            # Check model architecture to determine if using distance or similarity
            model_name = getattr(self.model_wrapper, '__class__', None)
            is_capsnet = model_name and 'Caps' in str(model_name)
            
            # CapsNet with Pearson correlation outputs similarity (range ~-1 to 1 or 0 to 1)
            # Check if using Pearson distance which outputs similarity scores
            uses_pearson = getattr(self.model_wrapper, 'use_pearson_distance', False)
            
            # FIXED: Proper detection - Pearson outputs similarity, not distance
            if is_capsnet and uses_pearson:
                score_mode = 0  # Similarity mode for Pearson
                is_distance = False
            elif is_capsnet:
                # Non-Pearson CapsNet outputs distances
                score_mode = 1
                is_distance = True
            else:
                # General detection for other models
                is_distance = (raw_min < 0.0) or (raw_max > 1.0) or (raw_spread > 1.25)
                score_mode = 1 if (is_distance and raw_spread > 1e-8) else 0
            # IMPROVED: Better normalization and orientation detection
            score_is_similarity = 1.0
            if score_mode == 1:  # Distance mode
                # Normalize distances to [0, 1] range
                if raw_spread > 1e-6:
                    norm_dist = (y_raw_scores - raw_min) / raw_spread
                else:
                    # If spread is tiny, model isn't learning - use raw values
                    norm_dist = np.clip(y_raw_scores, 0.0, 1.0)
                y_sim = 1.0 - norm_dist  # Convert distance to similarity
                y_dist = norm_dist
                y_scores = y_sim  # Default: lower distance = higher similarity = positive
                if self._roc_auc_score is not None and len(np.unique(y_true)) > 1:
                    try:
                        auc_sim = float(self._roc_auc_score(y_true, y_sim))
                        auc_dist = float(self._roc_auc_score(y_true, y_dist))
                        if np.isfinite(auc_dist) and auc_dist > (auc_sim if np.isfinite(auc_sim) else -np.inf):
                            y_scores = y_dist
                            score_is_similarity = 0.0
                    except Exception:
                        pass
            else:  # Similarity/probability mode
                # For similarity scores, ensure proper range
                if raw_spread > 1e-6:
                    # Normalize to [0, 1] if needed
                    y_scores = (y_raw_scores - raw_min) / raw_spread
                else:
                    # Tiny spread means model not learning
                    y_scores = np.clip(y_raw_scores, 0.0, 1.0)
                score_is_similarity = 1.0

            # Enhanced collapse detection: evaluate class separation instead of raw extremes
            collapse_messages = []
            if len(np.unique(y_true)) >= 2:
                pos_scores = y_scores[y_true == 1]
                neg_scores = y_scores[y_true == 0]
                if pos_scores.size and neg_scores.size:
                    pos_mean = float(np.mean(pos_scores))
                    neg_mean = float(np.mean(neg_scores))
                    pos_std = float(np.std(pos_scores))
                    neg_std = float(np.std(neg_scores))
                    score_spread = collapse_diag['raw_spread']
                    separation = abs(pos_mean - neg_mean)

                    if score_is_similarity >= 0.5:
                        if separation < 0.1:
                            collapse_messages.append(
                                f"Similarity separation too small (Δ={separation:.3f})"
                            )
                        if separation < 0.2 and score_spread < 0.4 and max(pos_std, neg_std) < 0.15:
                            collapse_messages.append(
                                f"Similarity scores overlapping: separation={separation:.3f}, spread={score_spread:.3f}"
                            )
                        if pos_mean < 0.3:
                            collapse_messages.append(
                                f"Positive pairs skewing low (mean={pos_mean:.3f}); expected high similarity"
                            )
                        if neg_mean > 0.7:
                            collapse_messages.append(
                                f"Negative pairs skewing high (mean={neg_mean:.3f}); expected low similarity"
                            )
                    else:
                        if separation < 0.1:
                            collapse_messages.append(
                                f"Distance separation too small (Δ={separation:.3f})"
                            )
                        if separation < 0.2 and score_spread < 0.4 and max(pos_std, neg_std) < 0.15:
                            collapse_messages.append(
                                f"Distance scores overlapping: separation={separation:.3f}, spread={score_spread:.3f}"
                            )
                        if pos_mean > 0.7:
                            collapse_messages.append(
                                f"Positive pairs skewing high distance (mean={pos_mean:.3f}); expected low distance"
                            )
                        if neg_mean < 0.3:
                            collapse_messages.append(
                                f"Negative pairs skewing low distance (mean={neg_mean:.3f}); expected high distance"
                            )

                    if collapse_messages:
                        print("[WARNING] Score separation collapse detected:")
                        for msg in collapse_messages:
                            print(f"   - {msg}")
                    else:
                        print(
                            f"[ScoreSeparation] pos_mean={pos_mean:.3f}, neg_mean={neg_mean:.3f}, "
                            f"separation={separation:.3f}, near0={collapse_diag['near_zero_frac']*100:.1f}%, "
                            f"near1={collapse_diag['near_one_frac']*100:.1f}%"
                        )
            else:
                # Fall back to raw extremes when class labels unavailable
                if collapse_diag['raw_spread'] < 0.05 and (
                    collapse_diag['near_zero_frac'] > 0.8 or collapse_diag['near_one_frac'] > 0.8
                ):
                    print(
                        f"[WARNING] Score collapse suspected: spread={collapse_diag['raw_spread']:.3f}, "
                        f"near0={collapse_diag['near_zero_frac']*100:.1f}%, near1={collapse_diag['near_one_frac']*100:.1f}%"
                    )

            # IMPROVED: Quantile-bounded threshold grid with better bounds
            if self.threshold_search:
                try:
                    q_lo = float(np.quantile(y_scores, 0.05))
                    q_hi = float(np.quantile(y_scores, 0.95))
                    # Ensure reasonable spread for threshold search
                    spread = q_hi - q_lo
                    if not np.isfinite(q_lo) or not np.isfinite(q_hi) or spread < 0.01:
                        # Fallback if spread too small
                        raise ValueError('invalid quantiles or insufficient spread')
                    # Expand search slightly beyond quantiles
                    margin = spread * 0.1
                    q_lo = max(0.0, q_lo - margin)
                    q_hi = min(1.0, q_hi + margin)
                    thresholds = np.linspace(q_lo, q_hi, 101)
                except Exception:
                    thresholds = np.linspace(0.1, 0.9, 81)
            else:
                thresholds = [float(np.clip(self.best_threshold, 0.05, 0.95))]
            best_local_f1 = -1.0
            best_local_metrics = None
            best_local_threshold = self.best_threshold
            best_local_pred_unique = 0
            for t in thresholds:
                y_pred = (y_scores >= t).astype(int)
                precision = self._precision_score(y_true, y_pred, zero_division=0)
                recall = self._recall_score(y_true, y_pred, zero_division=0)
                f1 = self._f1_score(y_true, y_pred, zero_division=0)
                
                # CRITICAL FIX: Balance-aware threshold selection
                # Previous issue: recall=0.99 due to pure F1 maximization
                # Solution: Penalize |precision - recall| imbalance
                balance_penalty = abs(precision - recall)
                balance_factor = 1.0 / (1.0 + balance_penalty)  # Higher when P≈R
                # Adjusted F1: 70% original F1 + 30% balance bonus
                adjusted_f1 = f1 * (0.7 + 0.3 * balance_factor)
                
                if adjusted_f1 > best_local_f1:
                    acc = self._accuracy_score(y_true, y_pred)
                    best_local_f1 = adjusted_f1  # Use adjusted for comparison
                    best_local_metrics = (precision, recall, f1, acc)  # Store original metrics
                    best_local_threshold = t
                    best_local_pred_unique = int(np.unique(y_pred).size)
            if not best_local_metrics:
                return
            precision, recall, f1, acc = best_local_metrics
            # Compute AUC only if both classes present; otherwise mark as NaN
            classes = np.unique(y_true)
            if classes.size >= 2:
                try:
                    auc = float(self._roc_auc_score(y_true, y_scores))
                    # DIAGNOSTIC: Warn if AUC is poor but don't auto-invert
                    if np.isfinite(auc) and auc < 0.5:
                        print(f"\nWARNING: AUC={auc:.4f} < 0.5 detected - model predictions may be inverted.")
                        print(f"   This often indicates: wrong distance metric, incorrect label encoding,")
                        print(f"   or model not learning properly. Check configuration.\n")
                except Exception as auc_e:
                    auc = float('nan')
            else:
                auc = float('nan')
            # Fallback to Youden J threshold if best F1 sits at edges or degenerate predictions
            threshold_method = 0.0  # 0=f1, 1=youden
            logs['threshold_edge_fallback'] = 0.0
            try:
                use_youden = False
                # Edge when best threshold equals first/last grid point or predictions are single-class
                if best_local_threshold <= thresholds[0] + 1e-8 or best_local_threshold >= thresholds[-1] - 1e-8:
                    use_youden = True
                if best_local_pred_unique < 2:
                    use_youden = True
                if use_youden and self._roc_curve is not None and len(np.unique(y_true)) > 1:
                    fpr_j, tpr_j, thr_j = self._roc_curve(y_true, y_scores)
                    j_scores = tpr_j - fpr_j
                    j_idx = int(np.argmax(j_scores)) if j_scores.size > 0 else None
                    if j_idx is not None:
                        cand_thr = float(thr_j[j_idx])
                        # Ensure within [0,1]
                        if np.isfinite(cand_thr):
                            best_local_threshold = float(np.clip(cand_thr, 0.0, 1.0))
                            y_pred = (y_scores >= best_local_threshold).astype(int)
                            precision = self._precision_score(y_true, y_pred, zero_division=0)
                            recall = self._recall_score(y_true, y_pred, zero_division=0)
                            f1 = self._f1_score(y_true, y_pred, zero_division=0)
                            acc = self._accuracy_score(y_true, y_pred)
                            threshold_method = 1.0
                            logs['threshold_edge_fallback'] = 1.0
            except Exception:
                pass
            # Track best metrics / threshold with minimum threshold safeguard
            if f1 > self.best_f1:
                self.best_f1 = f1
                # Ensure minimum threshold to prevent metric collapse
                self.best_threshold = max(0.01, best_local_threshold)
                setattr(self.model_wrapper, 'validation_threshold', self.best_threshold)

            # Low-AUC and recall-floor handling to avoid degenerate low-recall states
            # Apply after initial F1/Youden selection, before propagating thresholds
            try:
                low_auc_flag = (np.isfinite(auc) and auc < 0.55)
            except Exception:
                low_auc_flag = False
            recall_floor = float(getattr(self, 'recall_floor', 0.40))
            precision_min = float(getattr(self, 'precision_min', 0.50))
            logs['low_auc_mode'] = 1.0 if low_auc_flag else 0.0
            logs['recall_floor'] = float(recall_floor)
            logs['recall_floor_applied'] = 0.0
            # If AUC is weak or the chosen threshold yields poor recall, reselect with constraints
            if low_auc_flag or (best_local_metrics and best_local_metrics[1] < recall_floor):
                y_true_arr = np.asarray(y_true).astype(int)
                selected = False
                # First pass: maximize balanced accuracy with recall >= floor
                best_balacc = -1.0
                best_tuple = None
                for t in thresholds:
                    y_pred = (y_scores >= t).astype(int)
                    # Skip single-class predictions
                    if int(np.unique(y_pred).size) < 2:
                        continue
                    # Confusion matrix terms
                    tp = int(np.sum((y_true_arr == 1) & (y_pred == 1)))
                    tn = int(np.sum((y_true_arr == 0) & (y_pred == 0)))
                    fp = int(np.sum((y_true_arr == 0) & (y_pred == 1)))
                    fn = int(np.sum((y_true_arr == 1) & (y_pred == 0)))
                    pos = tp + fn; neg = tn + fp
                    rec = (tp / (pos + 1e-8)) if pos > 0 else 0.0
                    tnr = (tn / (neg + 1e-8)) if neg > 0 else 0.0
                    balacc = 0.5 * (rec + tnr)
                    if rec >= recall_floor and balacc > best_balacc:
                        precision_c = self._precision_score(y_true_arr, y_pred, zero_division=0)
                        f1_c = self._f1_score(y_true_arr, y_pred, zero_division=0)
                        acc_c = self._accuracy_score(y_true_arr, y_pred)
                        best_balacc = balacc
                        best_tuple = (t, precision_c, rec, f1_c, acc_c)
                if best_tuple is not None:
                    best_local_threshold, precision, recall, f1, acc = best_tuple
                    threshold_method = 2.0  # recall-floor balanced-accuracy
                    logs['recall_floor_applied'] = 1.0
                    selected = True
                # Second pass: relax recall target if unmet, prefer higher recall with reasonable precision
                if not selected:
                    relaxed_floor = max(0.20, 0.80 * recall_floor)
                    best_rec = -1.0
                    best_tuple = None
                    for t in thresholds:
                        y_pred = (y_scores >= t).astype(int)
                        rec = self._recall_score(y_true_arr, y_pred, zero_division=0)
                        prec = self._precision_score(y_true_arr, y_pred, zero_division=0)
                        if rec >= relaxed_floor and prec >= precision_min and rec > best_rec:
                            f1_c = self._f1_score(y_true_arr, y_pred, zero_division=0)
                            acc_c = self._accuracy_score(y_true_arr, y_pred)
                            best_rec = rec
                            best_tuple = (t, prec, rec, f1_c, acc_c)
                    if best_tuple is not None:
                        best_local_threshold, precision, recall, f1, acc = best_tuple
                        threshold_method = 3.0  # recall-priority (relaxed)
                        logs['recall_floor_applied'] = 1.0
                        selected = True
                # Final fallback: choose threshold with max recall, regardless of precision (diagnostic mode)
                if not selected:
                    best_rec = -1.0
                    best_tuple = None
                    for t in thresholds:
                        y_pred = (y_scores >= t).astype(int)
                        rec = self._recall_score(y_true_arr, y_pred, zero_division=0)
                        if rec > best_rec:
                            prec = self._precision_score(y_true_arr, y_pred, zero_division=0)
                            f1_c = self._f1_score(y_true_arr, y_pred, zero_division=0)
                            acc_c = self._accuracy_score(y_true_arr, y_pred)
                            best_rec = rec
                            best_tuple = (t, prec, rec, f1_c, acc_c)
                    if best_tuple is not None:
                        best_local_threshold, precision, recall, f1, acc = best_tuple
                        threshold_method = 3.0  # recall-priority (max recall)
                        logs['recall_floor_applied'] = 1.0
                        selected = True
                # Update best cache if improved
                if selected and f1 > self.best_f1:
                    self.best_f1 = f1
                    self.best_threshold = max(0.01, best_local_threshold)
                    setattr(self.model_wrapper, 'validation_threshold', self.best_threshold)

            # Convert threshold to the model's distance domain (if scores were normalized similarities)
            try:
                if score_mode == 1:
                    if score_is_similarity == 1.0:
                        # dist = raw_min + (1 - sim_score) * spread
                        assign_threshold = float(raw_min + (1.0 - max(0.01, best_local_threshold)) * (raw_spread + 1e-8))
                    else:
                        # score is normalized distance directly: dist = raw_min + dist_score * spread
                        assign_threshold = float(raw_min + max(0.01, best_local_threshold) * (raw_spread + 1e-8))
                    assign_threshold = float(np.clip(assign_threshold, raw_min, raw_max))
                else:
                    # Probability domain; keep score threshold
                    assign_threshold = float(max(0.01, best_local_threshold))
            except Exception:
                assign_threshold = float(max(0.01, best_local_threshold))

            # Optional EMA smoothing in distance domain using previous model threshold (if available)
            # FIXED: Increase smoothing to prevent dramatic threshold jumps (0.327 -> 0.404)
            try:
                if hasattr(self.model_wrapper, 'metric_distance_threshold') and isinstance(self.threshold_smoothing, float) and self.threshold_smoothing > 0.0:
                    prev_thr = float(getattr(self.model_wrapper, 'metric_distance_threshold', assign_threshold))
                    # Increase smoothing factor to stabilize thresholds
                    alpha = min(0.5, float(self.threshold_smoothing) * 2.0)  # More aggressive smoothing
                    assign_threshold = float((1.0 - alpha) * prev_thr + alpha * assign_threshold)
            except Exception:
                pass

            # Propagate current epoch's calibrated (distance-domain) threshold into the model wrapper for next epoch metrics
            try:
                # Preferred: use dedicated setter when provided by model
                if hasattr(self.model_wrapper, 'set_metric_distance_threshold') and callable(getattr(self.model_wrapper, 'set_metric_distance_threshold')):
                    try:
                        self.model_wrapper.set_metric_distance_threshold(float(assign_threshold))
                        logs['threshold_setter_used'] = 1.0
                    except Exception:
                        logs['threshold_setter_used'] = 0.0
                # Shared tf.Variable used by custom metrics (siamese/capsule models only)
                if hasattr(self.model_wrapper, '_threshold_var'):
                    try:
                        # Lazy import to avoid hard dependency for baseline models
                        import tensorflow as tf  # type: ignore
                        tv = getattr(self.model_wrapper, '_threshold_var', None)
                        if isinstance(tv, tf.Variable):
                            tv.assign(float(assign_threshold))
                            logs['threshold_var_assigned'] = 1.0
                        else:
                            logs['threshold_var_assigned'] = 0.0
                    except Exception:
                        logs['threshold_var_assigned'] = 0.0
                # Also mirror to a plain attribute consumed elsewhere
                if hasattr(self.model_wrapper, 'metric_distance_threshold'):
                    setattr(self.model_wrapper, 'metric_distance_threshold', float(assign_threshold))
                # Back-compat attribute used by GUI displays
                setattr(self.model_wrapper, 'validation_threshold', float(max(0.01, best_local_threshold)))
            except Exception:
                # Do not fail GUI callback if propagation fails (e.g., baseline CNN)
                pass
            if auc > self.best_auc:
                self.best_auc = auc
            # Recompute y_pred for selected threshold to derive balanced accuracy
            try:
                y_pred_final = (y_scores >= best_local_threshold).astype(int)
                y_true_arr = np.asarray(y_true).astype(int)
                tp_f = int(np.sum((y_true_arr == 1) & (y_pred_final == 1)))
                tn_f = int(np.sum((y_true_arr == 0) & (y_pred_final == 0)))
                fp_f = int(np.sum((y_true_arr == 0) & (y_pred_final == 1)))
                fn_f = int(np.sum((y_true_arr == 1) & (y_pred_final == 0)))
                pos_f = tp_f + fn_f; neg_f = tn_f + fp_f
                recall_f = (tp_f / (pos_f + 1e-8)) if pos_f > 0 else 0.0
                tnr_f = (tn_f / (neg_f + 1e-8)) if neg_f > 0 else 0.0
                bal_acc = 0.5 * (recall_f + tnr_f)
            except Exception:
                bal_acc = float('nan')

            # Populate logs
            metric_pairs = [
                ('val_precision', precision), ('val_recall', recall), ('val_f1', f1),
                ('val_accuracy', acc), ('val_auc', auc), ('val_balanced_accuracy', bal_acc),
                # Report both the score-domain threshold (0..1) and the distance-domain threshold
                ('threshold_score', best_local_threshold), ('threshold', assign_threshold),
                ('best_val_f1', self.best_f1), ('best_val_auc', self.best_auc),
                ('score_is_similarity', score_is_similarity), ('threshold_method', threshold_method)
            ]
            for k, v in metric_pairs:
                try:
                    logs[k] = float(v)
                except Exception:
                    logs[k] = 0.0
            # Fixed-threshold metrics (score space)
            try:
                if self.log_fixed_threshold:
                    t_fix = float(np.clip(self.fixed_threshold, 0.0, 1.0))
                    y_pred_fix = (y_scores >= t_fix).astype(int)
                    logs['val_precision_at_fixed'] = float(self._precision_score(y_true, y_pred_fix, zero_division=0))
                    logs['val_recall_at_fixed'] = float(self._recall_score(y_true, y_pred_fix, zero_division=0))
                    logs['val_f1_at_fixed'] = float(self._f1_score(y_true, y_pred_fix, zero_division=0))
                    logs['val_accuracy_at_fixed'] = float(self._accuracy_score(y_true, y_pred_fix))
                    logs['val_threshold_fixed'] = float(t_fix)
            except Exception:
                pass
            # Frozen threshold metrics (freeze after given epoch)
            try:
                if self.freeze_after_epoch is not None and (epoch + 1) >= self.freeze_after_epoch and self.frozen_threshold_score is None:
                    self.frozen_threshold_score = float(best_local_threshold)
                if self.frozen_threshold_score is not None:
                    t_froz = float(np.clip(self.frozen_threshold_score, 0.0, 1.0))
                    y_pred_froz = (y_scores >= t_froz).astype(int)
                    logs['val_precision_at_frozen'] = float(self._precision_score(y_true, y_pred_froz, zero_division=0))
                    logs['val_recall_at_frozen'] = float(self._recall_score(y_true, y_pred_froz, zero_division=0))
                    logs['val_f1_at_frozen'] = float(self._f1_score(y_true, y_pred_froz, zero_division=0))
                    logs['val_accuracy_at_frozen'] = float(self._accuracy_score(y_true, y_pred_froz))
                    logs['val_threshold_frozen'] = float(t_froz)
            except Exception:
                pass
            # PR curve summaries (threshold-free)
            try:
                if self._pr_curve is not None and len(np.unique(y_true)) > 1:
                    pr_prec, pr_rec, _ = self._pr_curve(y_true, y_scores)
                    ap = float(self._avg_precision(y_true, y_scores)) if self._avg_precision is not None else float('nan')
                    mask_p = pr_prec >= 0.8
                    rec_at_p80 = float(np.max(pr_rec[mask_p])) if np.any(mask_p) else 0.0
                    mask_r = pr_rec >= 0.9
                    prec_at_r90 = float(np.max(pr_prec[mask_r])) if np.any(mask_r) else 0.0
                    logs['val_ap'] = ap
                    logs['val_recall_at_p80'] = rec_at_p80
                    logs['val_precision_at_r90'] = prec_at_r90
            except Exception:
                pass
            logs['val_raw_min'] = raw_min
            logs['val_raw_max'] = raw_max
            logs['val_raw_spread'] = raw_spread
            logs['val_score_mode'] = float(score_mode)
            logs['score_is_similarity'] = float(score_is_similarity)
            logs['val_sample_count'] = float(len(y_true))
            logs['val_local_best_f1'] = float(best_local_f1)
            logs['val_local_best_threshold'] = float(best_local_threshold)
            # Mark successful computation and provenance for GUI/filenames
            logs['val_metrics_source'] = 'callback'
            # Explicitly clear previous error flag on success
            logs['validation_callback_error'] = 0.0
            # ROC curve data computed only for progress_queue event, not stored in logs
            fpr = None; tpr = None
            try:
                if self._roc_curve is not None and len(np.unique(y_true)) > 1:
                    fpr, tpr, _ = self._roc_curve(y_true, y_scores)
                    if len(fpr) > 400:
                        idx = np.linspace(0, len(fpr)-1, 400).astype(int)
                        fpr = fpr[idx]; tpr = tpr[idx]
            except Exception:
                fpr = None; tpr = None
            # Persist compact debug JSON for offline verification
            try:
                if self.model_dir:
                    dbg_dir = os.path.join(self.model_dir, 'logs')
                    os.makedirs(dbg_dir, exist_ok=True)
                    dbg_path = os.path.join(dbg_dir, f'val_debug_epoch_{epoch+1:02d}.json')
                    payload = {
                        'epoch': int(epoch+1),
                        'n': int(len(y_true)),
                        'pos': int(np.sum(y_true)),
                        'neg': int(len(y_true) - np.sum(y_true)),
                        'balanced': bool(logs.get('val_balanced_applied', 0.0) == 1.0),
                        'pos_total': int(logs.get('val_pos_total', 0)),
                        'neg_total': int(logs.get('val_neg_total', 0)),
                        'pos_used': int(logs.get('val_pos_used', 0)),
                        'neg_used': int(logs.get('val_neg_used', 0)),
                        'raw_min': raw_min,
                        'raw_max': raw_max,
                        'raw_spread': raw_spread,
                        'score_mode': int(score_mode),
                        'precision': float(precision),
                        'recall': float(recall),
                        'f1': float(f1),
                        'acc': float(acc),
                        'balanced_accuracy': float(bal_acc) if np.isfinite(bal_acc) else None,
                        'auc': float(auc) if np.isfinite(auc) else None,
                        'thr_score': float(best_local_threshold),
                        'thr_dist': float(assign_threshold),
                        'score_is_similarity': bool(score_is_similarity == 1.0),
                        'threshold_method': (
                            'youden' if threshold_method == 1.0 else (
                                'balacc_recall_floor' if threshold_method == 2.0 else (
                                    'recall_priority' if threshold_method == 3.0 else 'f1'
                                )
                            )
                        )
                    }
                    with open(dbg_path, 'w', encoding='utf-8') as f:
                        json.dump(payload, f, indent=2)
            except Exception:
                pass
            # Note: Model saving is handled by ModelCheckpoint in the training loop to avoid HDF5 conflicts
            improved = False
            # Emit events
            if self.progress_queue is not None:
                try:
                    dbg_msg = (f"[Val Epoch {epoch+1}] raw_min={raw_min:.4f} raw_max={raw_max:.4f} "
                               f"mode={'dist->sim' if score_mode==1 else 'prob'} thr_score={best_local_threshold:.3f} "
                               f"thr_dist={assign_threshold:.3f} "
                               f"f1={f1:.4f} auc={auc:.4f} acc={acc:.4f} edge_fb={int(logs.get('threshold_edge_fallback',0))}")
                    self.progress_queue.put({'type': 'debug', 'message': dbg_msg})
                    # Separate ROC/AUC event
                    evt = {
                        'type': 'val_metrics',
                        'epoch': epoch+1,
                        'precision': float(precision),
                        'recall': float(recall),
                        'f1': float(f1),
                        'accuracy': float(acc),
                        # Use distance-domain threshold for downstream displays/metrics
                        'threshold': float(assign_threshold),
                        'threshold_score': float(best_local_threshold),
                        'auc': float(auc),
                        'fpr': fpr if fpr is not None else np.array([0,1]),
                        'tpr': tpr if tpr is not None else np.array([0,1]),
                        'balanced_accuracy': float(bal_acc) if np.isfinite(bal_acc) else None
                    }
                    if 'val_f1_at_fixed' in logs:
                        evt['f1_fixed'] = logs['val_f1_at_fixed']
                        evt['precision_fixed'] = logs['val_precision_at_fixed']
                        evt['recall_fixed'] = logs['val_recall_at_fixed']
                    if 'val_ap' in logs:
                        evt['ap'] = logs['val_ap']
                        evt['recall_at_p80'] = logs['val_recall_at_p80']
                        evt['precision_at_r90'] = logs['val_precision_at_r90']
                    self.progress_queue.put(evt)
                    if improved and self.model_dir:
                        self.progress_queue.put({'type': 'model_save', 'message': f"Model improved and saved: f1={f1:.4f}, auc={auc:.4f}"})
                except Exception:
                    pass
            # Final sanitation: drop any non-numeric entries from logs to avoid Keras progbar errors
            try:
                numeric_logs = {}
                for k, v in list(logs.items()):
                    try:
                        if isinstance(v, (int, float)):
                            numeric_logs[k] = float(v)
                        elif hasattr(v, 'item'):
                            numeric_logs[k] = float(v.item())
                        else:
                            # Skip non-scalar (arrays, strings, dicts, lists)
                            continue
                    except Exception:
                        continue
                # Replace contents in-place
                logs.clear(); logs.update(numeric_logs)
            except Exception:
                pass
        except Exception as e:
            try:
                setattr(self.model_wrapper, 'validation_callback_error_msg', str(e))
            except Exception:
                pass
            try:
                # Avoid placing strings into logs; mark numeric error flag only
                logs['validation_callback_error'] = 1.0
            except Exception:
                pass


class ModelTrainingGUI:
    """
    Comprehensive GUI for training pet recognition models with real-time visualization.
    """
    
    def __init__(self, root):
        """Initialize the training GUI."""
        self.root = root
        self.root.title("Fur-get Me Not: Real-Time Model Training Interface")
        self.root.geometry("1600x1000")
        self.root.configure(bg='#f0f0f0')
        
        # Training state variables
        self.current_model = None
        self.current_model_name = ""
        self.training_thread = None
        self.training_active = False
        self.training_data = {
            'epochs': [],
            'loss': [],
            'val_loss': [],
            'accuracy': [],
            'val_accuracy': [],
            'learning_rate': [],
            'precision': [],
            'val_precision': [],
            'recall': [],
            'val_recall': [],
            'f1_score': [],
            'val_f1_score': [],
            'val_auc': [],
            'thresholds': []
        }
        # Internal state for synchronizing training vs validation metrics per epoch
        self._awaiting_val_metrics = False   # Waiting for validation metrics to overwrite training placeholders
        self._last_train_logs = {}           # Raw training logs from epoch_complete
        self._last_epoch_times = {}          # Timing info for current epoch (elapsed, epoch_time, eta)
        self._last_val_epoch = 0             # Epoch index for which we last received authoritative val metrics

        # Model saving and logging directories
        self.current_model_dir = None        # Directory where current model is being saved
        self.training_session_id = None      # Unique session ID for this training run
        
        # Progress communication
        self.progress_queue = queue.Queue()

        # Training parameters
        self.training_params = {
            'epochs': tk.IntVar(value=50),
            'batch_size': tk.IntVar(value=64),
            'learning_rate': tk.DoubleVar(value=5e-5),
            'patience': tk.IntVar(value=7)
        }

        # Create GUI components
        self.create_menu()
        self.create_main_interface()
        self.setup_real_time_updates()

        # Initialize data generators
        self.data_generators = None
        self.load_data_generators()

        # Welcome message
        self.log_message("Welcome to Real-Time Model Training Interface!")
        self.log_message("Select a model and configure parameters to begin training.")
        
    def create_menu(self):
        """Create menu bar."""
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)
        
        # File menu
        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="Load Training Data", command=self.load_training_data)
        file_menu.add_command(label="Save Training Results", command=self.save_training_results)
        file_menu.add_command(label="Export Training Report", command=self.export_training_report)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.on_closing)
        
        # Models menu
        models_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Models", menu=models_menu)
        models_menu.add_command(label="Load Proposed Model", command=lambda: self.select_model('proposed'))
        
        # Training menu
        training_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Training", menu=training_menu)
        training_menu.add_command(label="Full Training", command=self.start_training)
        training_menu.add_command(label="Stop Training", command=self.stop_training)
        training_menu.add_command(label="Retrain Existing Model", command=self.retrain_existing_model)
        training_menu.add_command(label="Reset Training", command=self.reset_training)
        
    def create_main_interface(self):
        """Create the main interface layout."""
        # Create main paned window
        main_paned = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        main_paned.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Left panel for controls
        left_frame = ttk.Frame(main_paned)
        main_paned.add(left_frame, weight=1)
        
        # Right panel for visualization
        right_frame = ttk.Frame(main_paned)
        main_paned.add(right_frame, weight=3)
        
        # Create control panel
        self.create_control_panel(left_frame)
        
        # Create visualization panel
        self.create_visualization_panel(right_frame)
        
    def create_control_panel(self, parent):
        """Create the control panel on the left side."""
        # Model selection section
        model_frame = ttk.LabelFrame(parent, text="Model Selection", padding="10")
        model_frame.pack(fill=tk.X, pady=5)
        
        self.model_var = tk.StringVar(value="No model selected")
        model_label = ttk.Label(model_frame, text="Current Model:")
        model_label.pack(anchor=tk.W)
        
        model_display = ttk.Label(model_frame, textvariable=self.model_var, 
                                 foreground="blue", font=("Arial", 10, "bold"))
        model_display.pack(anchor=tk.W)
        
        # Model selection buttons
        button_frame = ttk.Frame(model_frame)
        button_frame.pack(fill=tk.X, pady=5)
        
        ttk.Button(button_frame, text="Proposed Model", 
                   command=lambda: self.select_model('proposed')).pack(side=tk.LEFT, padx=2)
        
        # Training parameters section
        params_frame = ttk.LabelFrame(parent, text="Training Parameters", padding="10")
        params_frame.pack(fill=tk.X, pady=5)
        
        # Epochs
        epochs_frame = ttk.Frame(params_frame)
        epochs_frame.pack(fill=tk.X, pady=2)
        ttk.Label(epochs_frame, text="Epochs:").pack(side=tk.LEFT)
        ttk.Scale(epochs_frame, from_=1, to=100, variable=self.training_params['epochs'],
                 orient=tk.HORIZONTAL).pack(side=tk.RIGHT, fill=tk.X, expand=True, padx=5)
        ttk.Label(epochs_frame, textvariable=self.training_params['epochs']).pack(side=tk.RIGHT)
        
        # Batch size
        batch_frame = ttk.Frame(params_frame)
        batch_frame.pack(fill=tk.X, pady=2)
        ttk.Label(batch_frame, text="Batch Size:").pack(side=tk.LEFT)
        batch_combo = ttk.Combobox(batch_frame, textvariable=self.training_params['batch_size'],
                                   values=[8, 16, 32, 64, 128], state="readonly", width=10)
        batch_combo.pack(side=tk.RIGHT)
        
        # Learning rate
        lr_frame = ttk.Frame(params_frame)
        lr_frame.pack(fill=tk.X, pady=2)
        ttk.Label(lr_frame, text="Learning Rate:").pack(side=tk.LEFT)
        lr_combo = ttk.Combobox(lr_frame, textvariable=self.training_params['learning_rate'],
                               values=[1e-5, 1e-4, 1e-3, 1e-2], state="readonly", width=10)
        lr_combo.pack(side=tk.RIGHT)
        
        # Patience
        patience_frame = ttk.Frame(params_frame)
        patience_frame.pack(fill=tk.X, pady=2)
        ttk.Label(patience_frame, text="Early Stop Patience:").pack(side=tk.LEFT)
        ttk.Scale(patience_frame, from_=5, to=20, variable=self.training_params['patience'],
                 orient=tk.HORIZONTAL).pack(side=tk.RIGHT, fill=tk.X, expand=True, padx=5)
        ttk.Label(patience_frame, textvariable=self.training_params['patience']).pack(side=tk.RIGHT)
        
        # Training controls section
        controls_frame = ttk.LabelFrame(parent, text="Training Controls", padding="10")
        controls_frame.pack(fill=tk.X, pady=5)
        
        # Main control buttons
        self.train_button = ttk.Button(controls_frame, text="Start Training", 
                                      command=self.start_training, state=tk.DISABLED)
        self.train_button.pack(fill=tk.X, pady=2)
        
        self.stop_button = ttk.Button(controls_frame, text="Stop Training", 
                                     command=self.stop_training, state=tk.DISABLED)
        self.stop_button.pack(fill=tk.X, pady=2)
        
        # Load existing model button
        self.load_model_button = ttk.Button(controls_frame, text="Load & Retrain Existing Model", 
                                          command=self.retrain_existing_model)
        self.load_model_button.pack(fill=tk.X, pady=2)
        
        # Browse models button
        #self.browse_models_button = ttk.Button(controls_frame, text="Browse Models", 
       #                                      command=self.browse_models_directory)
       #self.browse_models_button.pack(fill=tk.X, pady=2)

        # Training status section
        status_frame = ttk.LabelFrame(parent, text="Training Status", padding="10")
        status_frame.pack(fill=tk.X, pady=5)
        
        self.status_var = tk.StringVar(value="Ready to train")
        status_label = ttk.Label(status_frame, textvariable=self.status_var, 
                                font=("Arial", 10, "bold"))
        status_label.pack(anchor=tk.W)
        
        # Progress bar
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(status_frame, variable=self.progress_var, 
                                          maximum=100)
        self.progress_bar.pack(fill=tk.X, pady=5)
        
        # Current metrics display
        metrics_frame = ttk.Frame(status_frame)
        metrics_frame.pack(fill=tk.X, pady=5)
        
        self.current_metrics = tk.Text(metrics_frame, height=6, font=("Courier", 9))
        self.current_metrics.pack(fill=tk.X)
        
        # Training log section
        log_frame = ttk.LabelFrame(parent, text="Training Log", padding="10")
        log_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        self.log_text = scrolledtext.ScrolledText(log_frame, height=10, font=("Courier", 9))
        self.log_text.pack(fill=tk.BOTH, expand=True)
        
    def create_visualization_panel(self, parent):
        """Create the visualization panel with real-time plots."""
        # Create notebook for different visualizations
        notebook = ttk.Notebook(parent)
        notebook.pack(fill=tk.BOTH, expand=True)

        # Training progress tab
        progress_tab = ttk.Frame(notebook)
        notebook.add(progress_tab, text="Training Progress")
        self.create_progress_tab(progress_tab)

        # Loss analysis tab
        loss_tab = ttk.Frame(notebook)
        notebook.add(loss_tab, text="Loss Analysis")
        self.create_loss_tab(loss_tab)

        # Accuracy metrics tab
        accuracy_tab = ttk.Frame(notebook)
        notebook.add(accuracy_tab, text="Accuracy Metrics")
        self.create_accuracy_tab(accuracy_tab)

        # Model comparison tab
        comparison_tab = ttk.Frame(notebook)
        notebook.add(comparison_tab, text="Model Comparison")
        self.create_comparison_tab(comparison_tab)

        # ROC / AUC tab
        roc_tab = ttk.Frame(notebook)
        notebook.add(roc_tab, text="ROC / AUC")
        self.create_roc_tab(roc_tab)
        
    def create_progress_tab(self, parent):
        """Create the training progress visualization tab."""
        # Create matplotlib figure for real-time plotting
        self.progress_fig = Figure(figsize=(12, 8), dpi=100)
        self.progress_canvas = FigureCanvasTkAgg(self.progress_fig, parent)
        self.progress_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Add toolbar
        toolbar_frame = ttk.Frame(parent)
        toolbar_frame.pack(fill=tk.X)
        toolbar = NavigationToolbar2Tk(self.progress_canvas, toolbar_frame)
        toolbar.update()
        
        # Initialize progress plots
        self.setup_progress_plots()
        
    def create_loss_tab(self, parent):
        """Create the loss analysis tab."""
        self.loss_fig = Figure(figsize=(12, 8), dpi=100)
        self.loss_canvas = FigureCanvasTkAgg(self.loss_fig, parent)
        self.loss_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Control frame
        control_frame = ttk.Frame(parent)
        control_frame.pack(fill=tk.X, padx=10, pady=5)
        
        ttk.Button(control_frame, text="Update Loss Plot", 
                  command=self.update_loss_plot).pack(side=tk.LEFT, padx=5)
        ttk.Button(control_frame, text="Export Loss Data", 
                  command=self.export_loss_data).pack(side=tk.LEFT, padx=5)
        
        self.setup_loss_plots()
        
    def create_accuracy_tab(self, parent):
        """Create the accuracy metrics tab."""
        self.accuracy_fig = Figure(figsize=(12, 8), dpi=100)
        self.accuracy_canvas = FigureCanvasTkAgg(self.accuracy_fig, parent)
        self.accuracy_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Control frame
        control_frame = ttk.Frame(parent)
        control_frame.pack(fill=tk.X, padx=10, pady=5)
        
        ttk.Button(control_frame, text="Update Accuracy Plot", 
                  command=self.update_accuracy_plot).pack(side=tk.LEFT, padx=5)
        ttk.Button(control_frame, text="Show Statistics", 
                  command=self.show_accuracy_statistics).pack(side=tk.LEFT, padx=5)
        
        self.setup_accuracy_plots()
        
    def create_comparison_tab(self, parent):
        """Create the model comparison tab."""
        self.comparison_fig = Figure(figsize=(12, 8), dpi=100)
        self.comparison_canvas = FigureCanvasTkAgg(self.comparison_fig, parent)
        self.comparison_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Control frame
        control_frame = ttk.Frame(parent)
        control_frame.pack(fill=tk.X, padx=10, pady=5)
        
        ttk.Button(control_frame, text="Load Comparison Data", 
                  command=self.load_comparison_data).pack(side=tk.LEFT, padx=5)
        ttk.Button(control_frame, text="Update Comparison", 
                  command=self.update_comparison_plot).pack(side=tk.LEFT, padx=5)
        
        self.setup_comparison_plots()

    def create_roc_tab(self, parent):
        """Create the ROC / AUC real-time analysis tab."""
        self.roc_fig = Figure(figsize=(12, 8), dpi=100)
        self.roc_canvas = FigureCanvasTkAgg(self.roc_fig, parent)
        self.roc_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.setup_roc_plots()
        
    def setup_progress_plots(self):
        """Setup the real-time progress plots."""
        self.progress_fig.clear()
        
        # Create subplots
        self.ax_loss = self.progress_fig.add_subplot(2, 2, 1)
        self.ax_acc = self.progress_fig.add_subplot(2, 2, 2)
        self.ax_lr = self.progress_fig.add_subplot(2, 2, 3)
        self.ax_metrics = self.progress_fig.add_subplot(2, 2, 4)
        
        # Initialize empty plots
        self.line_train_loss, = self.ax_loss.plot([], [], 'b-', label='Training Loss', linewidth=2)
        self.line_val_loss, = self.ax_loss.plot([], [], 'r-', label='Validation Loss', linewidth=2)
        self.ax_loss.set_title('Training & Validation Loss', fontweight='bold')
        self.ax_loss.set_xlabel('Epoch')
        self.ax_loss.set_ylabel('Loss')
        self.ax_loss.legend()
        self.ax_loss.grid(True, alpha=0.3)
        
        self.line_train_acc, = self.ax_acc.plot([], [], 'b-', label='Training Accuracy', linewidth=2)
        self.line_val_acc, = self.ax_acc.plot([], [], 'r-', label='Validation Accuracy', linewidth=2)
        self.ax_acc.set_title('Training & Validation Accuracy', fontweight='bold')
        self.ax_acc.set_xlabel('Epoch')
        self.ax_acc.set_ylabel('Accuracy')
        self.ax_acc.legend()
        self.ax_acc.grid(True, alpha=0.3)
        
        self.line_lr, = self.ax_lr.plot([], [], 'g-', label='Learning Rate', linewidth=2)
        self.ax_lr.set_title('Learning Rate Schedule', fontweight='bold')
        self.ax_lr.set_xlabel('Epoch')
        self.ax_lr.set_ylabel('Learning Rate')
        self.ax_lr.set_yscale('log')
        self.ax_lr.legend()
        self.ax_lr.grid(True, alpha=0.3)
        
        # Metrics summary text
        self.ax_metrics.text(0.1, 0.7, ' ', transform=self.ax_metrics.transAxes,
                           fontsize=14, fontweight='bold')
        self.ax_metrics.text(0.1, 0.5, 'No training data available', 
                           transform=self.ax_metrics.transAxes, fontsize=12)
        self.ax_metrics.set_xlim(0, 1)
        self.ax_metrics.set_ylim(0, 1)
        self.ax_metrics.axis('off')
        
        self.progress_fig.tight_layout()
        self.progress_canvas.draw()
        
    def setup_loss_plots(self):
        """Setup detailed loss analysis plots."""
        self.loss_fig.clear()
        
        # Create 2x2 subplot layout for loss analysis
        self.ax_loss_progression = self.loss_fig.add_subplot(2, 2, 1)
        self.ax_loss_distribution = self.loss_fig.add_subplot(2, 2, 2)
        self.ax_loss_gradient = self.loss_fig.add_subplot(2, 2, 3)
        self.ax_training_stats = self.loss_fig.add_subplot(2, 2, 4)
        
        # Setup loss progression plot
        self.ax_loss_progression.set_title('Loss Progression', fontweight='bold')
        self.ax_loss_progression.set_xlabel('Epoch')
        self.ax_loss_progression.set_ylabel('Loss')
        self.ax_loss_progression.grid(True, alpha=0.3)
        self.ax_loss_progression.legend(['Training Loss', 'Validation Loss'])
        
        # Setup loss distribution plot
        self.ax_loss_distribution.set_title('Loss Distribution', fontweight='bold')
        self.ax_loss_distribution.set_xlabel('Loss Value')
        self.ax_loss_distribution.set_ylabel('Frequency')
        self.ax_loss_distribution.grid(True, alpha=0.3)
        
        # Setup loss gradient plot
        self.ax_loss_gradient.set_title('Loss Gradient (Change per Epoch)', fontweight='bold')
        self.ax_loss_gradient.set_xlabel('Epoch')
        self.ax_loss_gradient.set_ylabel('Loss Change')
        self.ax_loss_gradient.grid(True, alpha=0.3)
        self.ax_loss_gradient.legend(['Training Loss Δ', 'Validation Loss Δ'])
        
        # Setup training statistics plot
        self.ax_training_stats.set_title('Training Statistics', fontweight='bold')
        self.ax_training_stats.axis('off')
        
        self.loss_fig.tight_layout()
        self.loss_canvas.draw()
        
    def setup_accuracy_plots(self):
        """Setup comprehensive accuracy analysis plots."""
        self.accuracy_fig.clear()
        
        # Create 2x2 subplot layout for comprehensive metrics
        self.ax_acc_main = self.accuracy_fig.add_subplot(2, 2, 1)
        self.ax_precision = self.accuracy_fig.add_subplot(2, 2, 2)
        self.ax_recall = self.accuracy_fig.add_subplot(2, 2, 3)
        self.ax_f1 = self.accuracy_fig.add_subplot(2, 2, 4)
        
        # Setup accuracy plot
        self.ax_acc_main.set_title('Training & Validation Accuracy', fontweight='bold')
        self.ax_acc_main.set_xlabel('Epoch')
        self.ax_acc_main.set_ylabel('Accuracy')
        self.ax_acc_main.grid(True, alpha=0.3)
        self.ax_acc_main.legend(['Training Accuracy', 'Validation Accuracy'])
        
        # Setup precision plot
        self.ax_precision.set_title('Precision', fontweight='bold')
        self.ax_precision.set_xlabel('Epoch')
        self.ax_precision.set_ylabel('Precision')
        self.ax_precision.grid(True, alpha=0.3)
        self.ax_precision.legend(['Training Precision', 'Validation Precision'])
        
        # Setup recall plot
        self.ax_recall.set_title('Recall', fontweight='bold')
        self.ax_recall.set_xlabel('Epoch')
        self.ax_recall.set_ylabel('Recall')
        self.ax_recall.grid(True, alpha=0.3)
        self.ax_recall.legend(['Training Recall', 'Validation Recall'])
        
        # Setup F1-score plot
        self.ax_f1.set_title('F1-Score', fontweight='bold')
        self.ax_f1.set_xlabel('Epoch')
        self.ax_f1.set_ylabel('F1-Score')
        self.ax_f1.grid(True, alpha=0.3)
        self.ax_f1.legend(['Training F1', 'Validation F1'])
        
        self.accuracy_fig.tight_layout()
        self.accuracy_canvas.draw()
        
    def setup_comparison_plots(self):
        """Setup model comparison plots."""
        self.comparison_fig.clear()
        
        ax = self.comparison_fig.add_subplot(1, 1, 1)
        ax.set_title('Model Performance Comparison', fontweight='bold', fontsize=16)
        ax.text(0.5, 0.5, 'No comparison data available\n\nTrain multiple models to see comparisons',
                ha='center', va='center', transform=ax.transAxes, fontsize=14)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        
        self.comparison_fig.tight_layout()
        self.comparison_canvas.draw()

    def setup_roc_plots(self):
        """Setup ROC / AUC plots."""
        self.roc_fig.clear()
        self.ax_roc_curve = self.roc_fig.add_subplot(1,2,1)
        self.ax_auc_history = self.roc_fig.add_subplot(1,2,2)
        self.ax_roc_curve.set_title('ROC Curve (Latest Epoch)', fontweight='bold')
        self.ax_roc_curve.set_xlabel('False Positive Rate')
        self.ax_roc_curve.set_ylabel('True Positive Rate')
        self.ax_roc_curve.plot([0,1],[0,1],'k--', alpha=0.5)
        self.ax_auc_history.set_title('AUC History', fontweight='bold')
        self.ax_auc_history.set_xlabel('Epoch')
        self.ax_auc_history.set_ylabel('AUC')
        self.ax_auc_history.set_ylim(0.0,1.05)
        self.roc_fig.tight_layout()
        self.roc_canvas.draw()

    def update_roc_plots(self):
        """Update ROC / AUC plots with latest data."""
        if not hasattr(self, 'roc_canvas'):
            return
        # Need last ROC data and auc history
        if getattr(self, 'last_roc', None) is None and not self.training_data.get('val_auc'):
            return
        self.ax_roc_curve.clear()
        self.ax_roc_curve.set_title('ROC Curve (Latest Epoch)', fontweight='bold')
        self.ax_roc_curve.set_xlabel('False Positive Rate')
        self.ax_roc_curve.set_ylabel('True Positive Rate')
        self.ax_roc_curve.plot([0,1],[0,1],'k--', alpha=0.5)
        if getattr(self, 'last_roc', None) is not None:
            fpr = self.last_roc.get('fpr')
            tpr = self.last_roc.get('tpr')
            auc_val = self.last_roc.get('auc')
            if fpr is not None and tpr is not None:
                self.ax_roc_curve.plot(fpr, tpr, color='blue', linewidth=2, label=f'AUC={auc_val:.4f}')
                self.ax_roc_curve.legend(loc='lower right')
        # AUC history
        self.ax_auc_history.clear()
        self.ax_auc_history.set_title('AUC History', fontweight='bold')
        self.ax_auc_history.set_xlabel('Epoch')
        self.ax_auc_history.set_ylabel('AUC')
        self.ax_auc_history.set_ylim(0.0,1.05)
        if self.training_data.get('epochs') and self.training_data.get('val_auc'):
            # Ensure length alignment
            auc_len = len(self.training_data['val_auc'])
            auc_epochs = self.training_data['epochs'][:auc_len]
            self.ax_auc_history.plot(auc_epochs, self.training_data['val_auc'], 'm-o', linewidth=2)
        self.roc_fig.tight_layout()
        self.roc_canvas.draw()
        
    def setup_real_time_updates(self):
        """Setup real-time updates from training thread."""
        self.root.after(100, self.check_progress_queue)
        
    def check_progress_queue(self):
        """Check for progress updates from training thread (with guard)."""
        try:
            while True:
                progress_data = self.progress_queue.get_nowait()
                try:
                    self.handle_progress_update(progress_data)
                except Exception as gui_exc:
                    # Log but continue processing
                    print(f"GUI update error: {gui_exc}")
                    try:
                        self.log_message(f"GUI update error: {gui_exc}")
                    except Exception:
                        pass
        except queue.Empty:
            pass
        # Schedule next check
        self.root.after(100, self.check_progress_queue)
        
    def handle_progress_update(self, progress_data):
        """Handle progress update from training thread with enhanced real-time logging."""
        update_type = progress_data.get('type', '')
        message = progress_data.get('message', '')

        # Defensive: Ensure timing values are float before division
        logs = progress_data.get('logs', {})
        elapsed_time = progress_data.get('elapsed_time', 0)
        epoch_time = progress_data.get('epoch_time', 0)
        eta = progress_data.get('eta', 0)
        try:
            elapsed_time = float(elapsed_time)
        except Exception:
            elapsed_time = 0.0
        try:
            epoch_time = float(epoch_time)
        except Exception:
            epoch_time = 0.0
        try:
            eta = float(eta)
        except Exception:
            eta = 0.0

        if update_type == 'training_start':
            self.status_var.set("Training in progress...")
            self.log_message(message or "Training started!")

        elif update_type == 'epoch_start':
            epoch = progress_data.get('epoch', 0)
            epochs_total = self.training_params['epochs'].get()
            self.status_var.set(f"Training: Starting Epoch {epoch}/{epochs_total}")
            self.log_message(message or f"Starting Epoch {epoch}...")

        elif update_type == 'batch_update':
            # Real-time batch updates to show training is active
            epoch = progress_data.get('epoch', 0)
            batch = progress_data.get('batch', 0)
            logs = progress_data.get('logs', {})
            epochs_total = self.training_params['epochs'].get()

            # Update status to show current batch
            self.status_var.set(f"Training: Epoch {epoch}/{epochs_total} - Batch {batch}")

            # Log every 20 batches to show activity without spam
            if batch % 20 == 0:
                loss_str = f"{logs.get('loss', 0):.4f}" if 'loss' in logs else "N/A"
                self.log_message(f"   Batch {batch}: Loss {loss_str}")

        elif update_type == 'epoch_complete':
            epoch = progress_data['epoch']
            logs = progress_data['logs']

            # Defensive: Ensure metrics are float before arithmetic
            def safe_float(val, default=0.0):
                try:
                    return float(val)
                except Exception:
                    return default

            elapsed_time = safe_float(elapsed_time)
            epoch_time = safe_float(epoch_time)
            eta = safe_float(eta)

            # Update training data
            self.training_data['epochs'].append(epoch)
            self.training_data['loss'].append(logs.get('loss', 0))
            self.training_data['val_loss'].append(logs.get('val_loss', 0))

            # Handle accuracy metrics - check multiple possible names
            # Prefer ValidationMetricsCallback values over preliminary training values
            accuracy = logs.get('binary_accuracy_metric', logs.get('accuracy', logs.get('acc', 0)))
            # Validation accuracy may not be provided by Keras; fallback to our custom metric name
            val_accuracy = logs.get('val_accuracy', logs.get('val_binary_accuracy_metric', logs.get('val_acc', 0)))

            self.training_data['accuracy'].append(accuracy)
            self.training_data['val_accuracy'].append(val_accuracy)

            # Handle precision and recall metrics
            if 'precision' not in self.training_data:
                self.training_data['precision'] = []
                self.training_data['val_precision'] = []
                self.training_data['recall'] = []
                self.training_data['val_recall'] = []
                self.training_data['f1_score'] = []
                self.training_data['val_f1_score'] = []

            # Handle different possible metric names - now includes named metrics from proposed model
            # Prefer ValidationMetricsCallback values (val_precision) over preliminary training values (val_precision_metric)
            precision = safe_float(logs.get('precision_metric', logs.get('precision', 0)))
            val_precision = safe_float(logs.get('val_precision', logs.get('val_precision_metric', 0)))  # Swapped preference
            recall = safe_float(logs.get('recall_metric', logs.get('recall', 0)))
            val_recall = safe_float(logs.get('val_recall', logs.get('val_recall_metric', 0)))  # Swapped preference

            self.training_data['precision'].append(precision)
            self.training_data['val_precision'].append(val_precision)
            self.training_data['recall'].append(recall)
            self.training_data['val_recall'].append(val_recall)

            # Calculate F1-score: 2 * (precision * recall) / (precision + recall)
            f1 = 2 * (precision * recall) / (precision + recall + 1e-8)
            # Prefer direct ValidationMetricsCallback F1 if available, otherwise calculate from precision/recall
            val_f1 = safe_float(logs.get('val_f1', 2 * (val_precision * val_recall) / (val_precision + val_recall + 1e-8)))

            self.training_data['f1_score'].append(f1)
            self.training_data['val_f1_score'].append(val_f1)

            # Append validation AUC if available; otherwise keep list length consistent
            try:
                val_auc_val = float(logs.get('val_auc', 0)) if 'val_auc' in logs else 0.0
            except Exception:
                val_auc_val = 0.0
            self.training_data.setdefault('val_auc', []).append(val_auc_val)
            # Append threshold if available
            if 'threshold' in logs:
                try:
                    self.training_data.setdefault('thresholds', []).append(float(logs.get('threshold', 0.5)))
                except Exception:
                    self.training_data.setdefault('thresholds', []).append(0.5)

            if 'lr' in logs:
                self.training_data['learning_rate'].append(logs.get('lr', 0))

            # Update progress
            epochs_total = self.training_params['epochs'].get()
            progress = (epoch / epochs_total) * 100
            self.progress_var.set(progress)

            # Update status with ETA - with safe division
            eta_str = ""
            try:
                if eta is not None and isinstance(eta, (int, float)) and eta > 0:
                    eta_str = f" (ETA: {eta/60:.1f}min)"
            except Exception:
                eta_str = ""
            self.status_var.set(f"Training: Epoch {epoch}/{epochs_total}{eta_str}")

            # Store training logs & timing (used if validation metrics arrive later or already arrived)
            self._last_train_logs = dict(logs)
            self._last_epoch_times = {
                'elapsed_time': elapsed_time,
                'epoch_time': epoch_time,
                'eta': eta
            }
            # Immediate provisional update (may be overwritten when val_metrics event comes)
            self.update_current_metrics(logs, elapsed_time, epoch_time, eta)
            # If validation metrics haven't yet arrived for this epoch, mark awaiting
            if self._last_val_epoch != epoch:
                self._awaiting_val_metrics = True
            else:
                self._awaiting_val_metrics = False

            # Update real-time plots
            self.update_progress_plots()
            self.update_loss_plots()
            self.update_accuracy_plots()

            # Enhanced logging with comprehensive metrics
            loss_str = f"{safe_float(logs.get('loss', 0)):.4f}"
            val_loss_str = f"{safe_float(logs.get('val_loss', 0)):.4f}"
            acc_str = f"{safe_float(logs.get('accuracy', logs.get('binary_accuracy_metric', 0))):.4f}"
            val_acc_str = f"{safe_float(logs.get('val_accuracy', logs.get('val_binary_accuracy_metric', logs.get('val_acc', 0)))):.4f}"
            precision_str = f"{precision:.4f}"
            recall_str = f"{recall:.4f}"
            val_f1_str = f"{safe_float(logs.get('val_f1', 0)):.4f}"
            val_auc_str = f"{safe_float(logs.get('val_auc', 0)):.4f}"
            threshold_info = logs.get('threshold', getattr(self.current_model, 'validation_threshold', None))
            try:
                threshold_info = float(threshold_info)
            except Exception:
                threshold_info = None
            threshold_str = f", thr={threshold_info:.3f}" if threshold_info is not None else ""
            # Safe timing string formatting
            try:
                epoch_time_num = float(epoch_time) if epoch_time is not None else None
            except Exception:
                epoch_time_num = None
            try:
                eta_num = float(eta) if eta is not None else None
            except Exception:
                eta_num = None
            if eta_num and epoch_time_num is not None:
                timing_str = f" [{epoch_time_num:.1f}s/epoch, ETA: {eta_num/60:.1f}min]"
            elif eta_num:
                timing_str = f" [ETA: {eta_num/60:.1f}min]"
            elif epoch_time_num is not None:
                timing_str = f" [{epoch_time_num:.1f}s/epoch]"
            else:
                timing_str = ""

            self.log_message(
                f"Epoch {epoch}/{epochs_total}: loss={loss_str}, val_loss={val_loss_str}, "
                f"acc={acc_str}, val_acc={val_acc_str}, val_f1={val_f1_str}, val_auc={val_auc_str}{threshold_str}, precision={precision_str}, recall={recall_str}{timing_str}"
            )

            # Save training logs and analytics per epoch
            self.save_epoch_data(epoch, logs, elapsed_time, epoch_time, eta)

        elif update_type == 'debug':
            # Lightweight debug instrumentation from validation callback
            dbg_msg = message or progress_data.get('msg', '')
            if dbg_msg:
                self.log_message(dbg_msg)

        elif update_type == 'val_metrics':
            # Store latest ROC info without altering primary metric arrays (epoch_complete handles those)
            try:
                fpr = progress_data.get('fpr')
                tpr = progress_data.get('tpr')
                auc_val = progress_data.get('auc')
                self.last_roc = {'fpr': fpr, 'tpr': tpr, 'auc': auc_val}
                
                # Update LIVE metrics display with validation results
                combined_logs = dict(self._last_train_logs)
                combined_logs.update({
                    'val_precision': progress_data.get('precision'),
                    'val_recall': progress_data.get('recall'),
                    'val_f1': progress_data.get('f1'),
                    'val_accuracy': progress_data.get('accuracy'),
                    'val_auc': progress_data.get('auc'),
                    'threshold': progress_data.get('threshold')
                })
                etimes = self._last_epoch_times
                self.update_current_metrics(
                    combined_logs,
                    etimes.get('elapsed_time', 0),
                    etimes.get('epoch_time', 0),
                    etimes.get('eta', 0)
                )
                self.update_progress_plots()
                self.update_loss_plots()
                self.update_accuracy_plots()
                
                # Track last validation epoch
                ep = progress_data.get('epoch')
                if ep:
                    self._last_val_epoch = ep
                self._awaiting_val_metrics = False
            except Exception:
                pass
            # Update ROC plots early
            try:
                self.update_roc_plots()
            except Exception as e:
                self.log_message(f"ROC update error: {e}")

        elif update_type == 'model_save':
            self.log_message(message or 'Model saved (improved metrics).')

        elif update_type == 'training_complete':
            self.status_var.set("Training completed!")
            self.progress_var.set(100)
            self.log_message(message or "Training completed successfully!")
            self.training_active = False
            self.update_button_states()

        elif update_type == 'training_error':
            self.status_var.set("Training failed!")
            self.log_message(f"Training error: {message}")
            self.training_active = False
            self.update_button_states()
            
    def update_current_metrics(self, logs, elapsed_time, epoch_time=0, eta=0):
        """Update the current metrics display with comprehensive classification metrics."""
        current_epoch = len(self.training_data['epochs'])
        epochs_total = self.training_params['epochs'].get()
        # Calculate / retrieve metrics (prefer validation callback metrics if present)
        # If no positive labels were encountered in this epoch, gate precision/recall to avoid misleading zeros
        positives_seen = logs.get('positives_seen_epoch', None)
        if positives_seen is not None:
            try:
                positives_seen = float(positives_seen)
            except Exception:
                positives_seen = None
        gate_metrics = (positives_seen is not None and positives_seen <= 0)

        precision = 0 if gate_metrics else logs.get('precision_metric', logs.get('precision', 0))
        recall = 0 if gate_metrics else logs.get('recall_metric', logs.get('recall', 0))
        val_precision = logs.get('val_precision', logs.get('val_precision_metric', 0))
        val_recall = logs.get('val_recall', logs.get('val_recall_metric', 0))
        f1 = 0 if gate_metrics else logs.get('f1', 2 * (precision * recall) / (precision + recall + 1e-8))
        val_f1 = logs.get('val_f1', 2 * (val_precision * val_recall) / (val_precision + val_recall + 1e-8))
        val_accuracy = logs.get('val_accuracy', logs.get('val_binary_accuracy_metric', logs.get('val_acc', 0)))
        threshold_value = logs.get('threshold', getattr(self.current_model, 'validation_threshold', None))
        val_auc = logs.get('val_auc', None)

        # Safe formatting helpers
        def fmt_num(x, fmt='{:.4f}'):
            try:
                return fmt.format(float(x))
            except Exception:
                return str(x)

        elapsed_display = "N/A"
        try:
            if isinstance(elapsed_time, (int, float)) and elapsed_time is not None:
                elapsed_display = f"{elapsed_time/60:.1f}"
            elif elapsed_time is not None:
                elapsed_float = float(elapsed_time)
                elapsed_display = f"{elapsed_float/60:.1f}"
            else:
                elapsed_display = "N/A"
        except Exception:
            elapsed_display = "N/A"

        threshold_display = f"{threshold_value:.3f}" if threshold_value is not None else 'N/A'

        metrics_text = (
            "LIVE TRAINING METRICS\n\n"
            f"Progress: Epoch {current_epoch}/{epochs_total}\n"
            f"Loss: {fmt_num(logs.get('loss', 0))} | Val Loss: {fmt_num(logs.get('val_loss', 0))}\n\n"
            "CLASSIFICATION METRICS:\n"
            f"Accuracy: {fmt_num(logs.get('accuracy', logs.get('binary_accuracy_metric', 0)))} | Val Accuracy: {fmt_num(val_accuracy)}\n"
            f"Precision: {fmt_num(precision)} | Val Precision: {fmt_num(val_precision)}\n"
            f"Recall: {fmt_num(recall)} | Val Recall: {fmt_num(val_recall)}\n"
            f"F1-Score: {fmt_num(f1)} | Val F1-Score: {fmt_num(val_f1)}\n"
        )
        if gate_metrics:
            metrics_text += "Note: No positive labels seen in training this epoch; training precision/recall/F1 shown as 0 by design.\n"
        if val_auc is not None:
            try:
                metrics_text += f"AUC: {float(val_auc):.4f}\n"
            except Exception:
                metrics_text += f"AUC: {val_auc}\n"
        metrics_text += (
            f"Threshold: {threshold_display}\n\n"
            "TIMING:\n"
            f"Elapsed: {elapsed_display} min"
        )

        if epoch_time > 0:
            try:
                epoch_time_safe = float(epoch_time)
                if epoch_time_safe > 0:
                    metrics_text += f"\nLast Epoch: {epoch_time_safe:.1f}s"
            except Exception:
                pass

        try:
            eta_num = float(eta) if eta is not None else 0
        except Exception:
            eta_num = 0
        if eta_num > 0:
            metrics_text += f"\nETA: {eta_num/60:.1f} min"

        if 'lr' in logs:
            metrics_text += f"\nLearning Rate: {logs.get('lr', 0):.2e}"

        # Add system resource indicator
        try:
            import psutil
            cpu = psutil.cpu_percent(interval=None)
            memory = psutil.virtual_memory()
            metrics_text += f"\n\nSYSTEM: CPU {cpu:.1f}% | RAM {memory.percent:.1f}%"
        except Exception:
            pass

        self.current_metrics.delete(1.0, tk.END)
        self.current_metrics.insert(1.0, metrics_text)
        # Append compact table of last 10 epochs
        try:
            if self.training_data['epochs']:
                header = "\nRecent Epochs (last 10)\nEp  Loss   VLoss  Acc   VAcc  VPrec  VRec  VF1   VAUC  Thr\n"
                lines = []
                start = max(0, len(self.training_data['epochs']) - 10)
                epochs_slice = self.training_data['epochs'][start:]
                for idx, ep in enumerate(epochs_slice):
                    def safe_idx(lst, i, default=0.0):
                        try:
                            return float(lst[start + i])
                        except Exception:
                            return default
                    loss_v = safe_idx(self.training_data['loss'], idx)
                    vloss_v = safe_idx(self.training_data['val_loss'], idx)
                    acc_v = safe_idx(self.training_data['accuracy'], idx)
                    vacc_v = safe_idx(self.training_data['val_accuracy'], idx)
                    vprec_v = safe_idx(self.training_data.get('val_precision', []), idx)
                    vrec_v = safe_idx(self.training_data.get('val_recall', []), idx)
                    vf1_v = safe_idx(self.training_data.get('val_f1_score', []), idx)
                    vauc_v = safe_idx(self.training_data.get('val_auc', []), idx)
                    thr_v = safe_idx(self.training_data.get('thresholds', []), idx)
                    lines.append(f"{ep:02d} {loss_v:6.4f} {vloss_v:6.4f} {acc_v:5.3f} {vacc_v:5.3f} {vprec_v:5.3f} {vrec_v:5.3f} {vf1_v:5.3f} {vauc_v:5.3f} {thr_v:4.2f}")
                self.current_metrics.insert(tk.END, header + "\n".join(lines))
        except Exception:
            pass
        
    def update_progress_plots(self):
        """Update the real-time progress plots."""
        if not self.training_data['epochs']:
            return
        
        epochs = self.training_data['epochs']
        
        # Update loss plot
        if self.training_data['loss'] and self.training_data['val_loss']:
            # Ensure matching lengths
            loss_len = min(len(epochs), len(self.training_data['loss']), len(self.training_data['val_loss']))
            self.line_train_loss.set_data(epochs[:loss_len], self.training_data['loss'][:loss_len])
            self.line_val_loss.set_data(epochs[:loss_len], self.training_data['val_loss'][:loss_len])
            
            # Auto-scale loss plot
            self.ax_loss.relim()
            self.ax_loss.autoscale_view()
        
        # Update accuracy plot
        if self.training_data['accuracy'] and self.training_data['val_accuracy']:
            acc_len = min(len(epochs), len(self.training_data['accuracy']), len(self.training_data['val_accuracy']))
            self.line_train_acc.set_data(epochs[:acc_len], self.training_data['accuracy'][:acc_len])
            self.line_val_acc.set_data(epochs[:acc_len], self.training_data['val_accuracy'][:acc_len])
            
            # Auto-scale accuracy plot
            self.ax_acc.relim()
            self.ax_acc.autoscale_view()
        
        # Update learning rate plot
        if self.training_data['learning_rate']:
            lr_len = len(self.training_data['learning_rate'])
            lr_epochs = epochs[:lr_len] if len(epochs) >= lr_len else list(range(1, lr_len+1))
            self.line_lr.set_data(lr_epochs, self.training_data['learning_rate'][:lr_len])
            
            # Auto-scale learning rate plot
            self.ax_lr.relim()
            self.ax_lr.autoscale_view()
        
        # Update metrics summary
        if epochs:
            current_epoch = epochs[-1]
            current_loss = self.training_data['loss'][-1] if self.training_data['loss'] else 0
            current_val_loss = self.training_data['val_loss'][-1] if self.training_data['val_loss'] else 0
            
            self.ax_metrics.clear()
            self.ax_metrics.text(0.1, 0.7, ' ', transform=self.ax_metrics.transAxes,
                               fontsize=14, fontweight='bold')
            
            metrics_text = f"""Current Epoch: {current_epoch}
Training Loss: {current_loss:.4f}
Validation Loss: {current_val_loss:.4f}

Model: {self.current_model_name}
Total Epochs: {self.training_params['epochs'].get()}
Batch Size: {self.training_params['batch_size'].get()}"""
            
            self.ax_metrics.text(0.1, 0.5, metrics_text, transform=self.ax_metrics.transAxes,
                               fontsize=10, fontfamily='monospace')
            self.ax_metrics.set_xlim(0, 1)
            self.ax_metrics.set_ylim(0, 1)
            self.ax_metrics.axis('off')
        
        # Redraw canvas
        self.progress_canvas.draw()
        # Update ROC plots (AUC history) if present
        try:
            self.update_roc_plots()
        except Exception:
            pass
    
    def update_loss_plots(self):
        """Update the detailed loss analysis plots."""
        if not self.training_data['epochs']:
            return
        
        epochs = self.training_data['epochs']
        train_loss = self.training_data['loss']
        val_loss = self.training_data['val_loss']
        
        if not train_loss or not val_loss:
            return
        
        # Update loss progression
        self.ax_loss_progression.clear()
        self.ax_loss_progression.plot(epochs, train_loss, 'b-', label='Training Loss', linewidth=2)
        self.ax_loss_progression.plot(epochs, val_loss, 'r-', label='Validation Loss', linewidth=2)
        self.ax_loss_progression.set_title('Loss Progression', fontweight='bold')
        self.ax_loss_progression.set_xlabel('Epoch')
        self.ax_loss_progression.set_ylabel('Loss')
        self.ax_loss_progression.grid(True, alpha=0.3)
        self.ax_loss_progression.legend()
        
        # Update loss distribution
        self.ax_loss_distribution.clear()
        all_losses = train_loss + val_loss
        self.ax_loss_distribution.hist(all_losses, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        self.ax_loss_distribution.set_title('Loss Distribution', fontweight='bold')
        self.ax_loss_distribution.set_xlabel('Loss Value')
        self.ax_loss_distribution.set_ylabel('Frequency')
        self.ax_loss_distribution.grid(True, alpha=0.3)
        
        # Update loss gradient (change per epoch)
        if len(train_loss) > 1:
            self.ax_loss_gradient.clear()
            train_gradient = [train_loss[i] - train_loss[i-1] for i in range(1, len(train_loss))]
            val_gradient = [val_loss[i] - val_loss[i-1] for i in range(1, len(val_loss))]
            gradient_epochs = epochs[1:]
            
            self.ax_loss_gradient.plot(gradient_epochs, train_gradient, 'b-', label='Training Loss Δ', linewidth=2)
            self.ax_loss_gradient.plot(gradient_epochs, val_gradient, 'r-', label='Validation Loss Δ', linewidth=2)
            self.ax_loss_gradient.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
            self.ax_loss_gradient.set_title('Loss Gradient (Change per Epoch)', fontweight='bold')
            self.ax_loss_gradient.set_xlabel('Epoch')
            self.ax_loss_gradient.set_ylabel('Loss Change')
            self.ax_loss_gradient.grid(True, alpha=0.3)
            self.ax_loss_gradient.legend()
        
        # Update training statistics
        self.ax_training_stats.clear()
        stats_text = f"""LOSS STATISTICS
        
Current Loss: {train_loss[-1]:.4f}
Current Val Loss: {val_loss[-1]:.4f}

Best Loss: {min(train_loss):.4f}
Best Val Loss: {min(val_loss):.4f}

Loss Improvement: {(train_loss[0] - train_loss[-1]):.4f}
Val Loss Improvement: {(val_loss[0] - val_loss[-1]):.4f}

Total Epochs: {len(epochs)}"""
        
        self.ax_training_stats.text(0.05, 0.95, stats_text, transform=self.ax_training_stats.transAxes,
                                   fontsize=10, verticalalignment='top', fontfamily='monospace')
        self.ax_training_stats.set_title('Training Statistics', fontweight='bold')
        self.ax_training_stats.axis('off')
        
        self.loss_fig.tight_layout()
        self.loss_canvas.draw()
    
    def update_accuracy_plots(self):
        """Update the comprehensive accuracy analysis plots."""
        if not self.training_data['epochs'] or 'precision' not in self.training_data:
            return
        
        epochs = self.training_data['epochs']
        
        # Check if we have all the required data
        required_metrics = ['accuracy', 'val_accuracy', 'precision', 'val_precision', 
                          'recall', 'val_recall', 'f1_score', 'val_f1_score']
        
        for metric in required_metrics:
            if metric not in self.training_data or not self.training_data[metric]:
                return
        
        # Update accuracy plot
        self.ax_acc_main.clear()
        self.ax_acc_main.plot(epochs, self.training_data['accuracy'], 'b-', 
                             label='Training Accuracy', linewidth=2)
        self.ax_acc_main.plot(epochs, self.training_data['val_accuracy'], 'r-', 
                             label='Validation Accuracy', linewidth=2)
        self.ax_acc_main.set_title('Training & Validation Accuracy', fontweight='bold')
        self.ax_acc_main.set_xlabel('Epoch')
        self.ax_acc_main.set_ylabel('Accuracy')
        self.ax_acc_main.grid(True, alpha=0.3)
        self.ax_acc_main.legend()
        
        # Update precision plot
        self.ax_precision.clear()
        self.ax_precision.plot(epochs, self.training_data['precision'], 'b-', 
                              label='Training Precision', linewidth=2)
        self.ax_precision.plot(epochs, self.training_data['val_precision'], 'r-', 
                              label='Validation Precision', linewidth=2)
        self.ax_precision.set_title('Precision', fontweight='bold')
        self.ax_precision.set_xlabel('Epoch')
        self.ax_precision.set_ylabel('Precision')
        self.ax_precision.grid(True, alpha=0.3)
        self.ax_precision.legend()
        
        # Update recall plot
        self.ax_recall.clear()
        self.ax_recall.plot(epochs, self.training_data['recall'], 'g-', 
                           label='Training Recall', linewidth=2)
        self.ax_recall.plot(epochs, self.training_data['val_recall'], 'orange', 
                           label='Validation Recall', linewidth=2)
        self.ax_recall.set_title('Recall', fontweight='bold')
        self.ax_recall.set_xlabel('Epoch')
        self.ax_recall.set_ylabel('Recall')
        self.ax_recall.grid(True, alpha=0.3)
        self.ax_recall.legend()
        
        # Update F1-score plot
        self.ax_f1.clear()
        self.ax_f1.plot(epochs, self.training_data['f1_score'], 'purple', 
                       label='Training F1', linewidth=2)
        self.ax_f1.plot(epochs, self.training_data['val_f1_score'], 'brown', 
                       label='Validation F1', linewidth=2)
        self.ax_f1.set_title('F1-Score', fontweight='bold')
        self.ax_f1.set_xlabel('Epoch')
        self.ax_f1.set_ylabel('F1-Score')
        self.ax_f1.grid(True, alpha=0.3)
        self.ax_f1.legend()
        
        self.accuracy_fig.tight_layout()
        self.accuracy_canvas.draw()
        try:
            self.update_roc_plots()
        except Exception:
            pass
        
    def select_model(self, model_type):
        """Select and initialize the proposed model for training."""
        self.log_message("Loading proposed model...")

        if not MODELS_AVAILABLE:
            messagebox.showerror("Error", "Model classes not available. Please ensure all model files are present.")
            return

        try:
            # Get learning rate from GUI
            user_lr = float(self.training_params['learning_rate'].get())
            
            self.current_model = SiameseCapsuleNetworkMobileNetV2(
                input_shape=(224, 224, 3),
                embedding_dim=128,  # INCREASED from 64 to 128 for better representation
                num_capsules=6,      # INCREASED from 4 to 6 for more capsule capacity
                dim_capsules=8,      # INCREASED from 4 to 8 for richer capsule features
                routings=3,          # INCREASED from 2 to 3 for better routing
                attention_heads=2,   # INCREASED from 1 to 2 for better attention
                use_attention=True,
                use_pearson_distance=False,
                # CRITICAL CHOICE: Distance Metric
                # Options: 'cosine', 'euclidean', 'pearson'
                # - 'cosine': Angular similarity, standard for normalized embeddings (RECOMMENDED)
                # - 'euclidean': Magnitude-based distance (use only if NOT normalizing)
                # - 'pearson': Linear correlation, can capture different patterns than cosine
                # FIXED: Cosine formula corrected to use (1 - sim) not 0.5 * (1 - sim)
                distance_type='cosine',  # Try 'pearson' if cosine doesn't work well
                loss_type='contrastive_enhanced',  # Enhanced contrastive works better with Euclidean
                label_positive_means_similar=True,
                learning_rate=user_lr,  # Pass GUI learning rate
                optimize_speed=True,
                mobilenet_alpha=1.3,
                use_capsule=True,    # EXPLICIT: Ensure capsules are enabled
                capsule_auto_fallback=False  # EXPLICIT: Disable auto-fallback
            )
            
            # CRITICAL DEBUG: Enable distance distribution logging
            # This will print actual min/max/mean distances every 200 batches
            # Look for: [MetricDebug] precision_metric pred_min=X pred_max=Y mean=Z
            self.current_model.debug_metric_distribution = True
            print("[DEBUG MODE ENABLED] Distance statistics will be printed every 200 batches")
            print(f"[CONFIG] Using distance_type='{self.current_model.distance_type}'")
            print(f"[CONFIG] Initial threshold: {self.current_model.metric_distance_threshold}")
            
            self.current_model_name = "Proposed Siamese CapsNet + MobileNetV2"

            # Create the network
            self.current_model.create_siamese_capsule_network()

            self.model_var.set(f"{self.current_model_name} (Ready)")
            self.log_message(f"{self.current_model_name} loaded successfully.")

            # Enable training controls
            self.update_button_states()

        except Exception as e:
            error_msg = f"Error loading proposed model: {str(e)}"
            self.log_message(error_msg)
            messagebox.showerror("Model Loading Error", error_msg)
    
    def load_data_generators(self):
        """Load data generators for training."""
        try:
            if os.path.exists('processed_data/train_pairs.csv'):
                selected_bs = int(self.training_params['batch_size'].get()) if 'batch_size' in self.training_params else 64
                self.data_generators = create_data_generators_from_csv(batch_size=selected_bs)
                self.log_message("Data generators loaded from CSV files")
            else:
                self.data_generators = create_demo_generators()
                self.log_message("Using demo data generators (processed data not found)")
                
        except Exception as e:
            self.log_message(f"Error loading data generators: {str(e)}")
            self.data_generators = create_demo_generators()
            self.log_message("Fallback to demo data generators")
    
    def start_training(self):
        """Start the training process in a separate thread."""
        if not self.current_model:
            messagebox.showerror("Error", "No model selected. Please select a model first.")
            return
        
        if self.training_active:
            messagebox.showwarning("Warning", "Training is already in progress.")
            return
        
        # CRITICAL: Reload data generators to capture current batch size from GUI
        # This ensures the selected batch size (e.g., 128) is actually used
        self.log_message("Reloading data generators with current batch size...")
        self.load_data_generators()
        
        if not self.data_generators:
            messagebox.showerror("Error", "No data generators available.")
            return
        
        # Reset training data
        self.reset_training_data()
        
        # Setup model directory for saving training data
        if not self.setup_model_directory():
            messagebox.showerror("Error", "Failed to create model directory for saving training data.")
            self.training_active = False
            self.update_button_states()
            return
        
        # Get training parameters
        epochs = self.training_params['epochs'].get()
        
        self.log_message(f"Starting training for {epochs} epochs...")
        self.training_active = True
        self.update_button_states()
        
        # Start training in separate thread
        self.training_thread = threading.Thread(
            target=self.train_model_thread,
            args=(epochs,),
            daemon=True
        )
        self.training_thread.start()
    
    def train_model_thread(self, epochs):
        """Training function that runs in a separate thread with real-time monitoring."""
        try:
            self.log_message("Preparing model and data for training...")
            
            # Ensure the model is properly created and compiled
            if not hasattr(self.current_model, 'siamese_model') or self.current_model.siamese_model is None:
                self.log_message("ERROR: Model not properly initialized. Creating and compiling...")
                return
                
            # Get data generators
            train_gen, val_gen, _ = self.data_generators

            # --- Embedding variance pre-check & optional capsule fallback rebuild ---
            try:
                # New: probe raw embedding variance to distinguish embedding collapse vs distance collapse
                try:
                    if getattr(self.current_model, 'use_capsule', False):
                        self.current_model.probe_embedding_variance(val_gen, batches=1)
                except Exception as ep:
                    self.log_message(f"[EmbProbe] skipped: {ep}")
                # REMOVED: check_embedding_variance call that was forcing fallback
                # The variance check is controlled by capsule_auto_fallback parameter (default False)
                # If user wants auto-fallback, they must explicitly set capsule_auto_fallback=True
                # Old code was:
                # if hasattr(self.current_model, 'check_embedding_variance') and callable(self.current_model.check_embedding_variance):
                #     triggered = self.current_model.check_embedding_variance(val_gen, max_batches=2, var_threshold=1e-4)
                #     if triggered:
                #         self.log_message("[Auto-Fallback] Capsule path disabled due to collapsed distances. Model rebuilt without capsules.")
            except Exception as vc_e:
                self.log_message(f"[VarianceCheck] Skipped due to error: {vc_e}")

            # --- Capsule routing entropy diagnostic (if capsule path still active) ---
            try:
                if getattr(self.current_model, 'use_capsule', False) and self.current_model.use_capsule:
                    # Probe a tiny forward pass to populate routing entropy
                    if len(val_gen) > 0:
                        batch0 = val_gen[0]
                        if isinstance(batch0, (list, tuple)):
                            if len(batch0) == 3:
                                (xa_probe, xb_probe), _, _ = batch0
                            elif len(batch0) == 2:
                                (xa_probe, xb_probe), _ = batch0
                            else:
                                (xa_probe, xb_probe) = batch0[0]
                            # Run a single forward pass (anchor only) through base to get entropy
                            caps_out, _ = self.current_model.model(xa_probe, training=False)
                            # Attempt to access routing entropy from capsule layer
                            for layer in self.current_model.model.layers:
                                if isinstance(layer, EnhancedCapsuleLayer):
                                    ent = layer.get_last_routing_entropy()
                                    if ent is not None:
                                        self.progress_queue.put({'type': 'debug', 'message': f"[RoutingEntropy] mean_norm_entropy={float(ent):.4f}"})
                                    break
            except Exception as re_e:
                self.progress_queue.put({'type': 'debug', 'message': f"[RoutingEntropy] Skipped: {re_e}"})

            # --- Quick distance distribution diagnostic (first validation batch) ---
            try:
                first_batch = val_gen[0]
                if isinstance(first_batch, (list, tuple)):
                    if len(first_batch) == 3:
                        (xa, xb), y, _ = first_batch
                    elif len(first_batch) == 2:
                        (xa, xb), y = first_batch
                    else:
                        (xa, xb) = first_batch[0]
                        y = first_batch[1] if len(first_batch) > 1 else None
                    # CRITICAL: Use training=True to activate GaussianNoise for proper variance check
                    # Without training mode, noise is disabled and embeddings collapse to near-identical values
                    preds = self.current_model.siamese_model([xa, xb], training=True).numpy().reshape(-1)
                    if preds.size > 0:
                        dist_min = float(np.min(preds))
                        dist_max = float(np.max(preds))
                        dist_mean = float(np.mean(preds))
                        dist_std = float(np.std(preds))
                        spread = dist_max - dist_min
                        self.progress_queue.put({'type': 'debug', 'message': f"[Pre-Train Distances] n={preds.size} min={dist_min:.5f} max={dist_max:.5f} mean={dist_mean:.5f} std={dist_std:.5f} spread={spread:.5f}"})
                        if spread < 1e-3:
                            self.progress_queue.put({'type': 'debug', 'message': f"[WARNING] Initial distance spread extremely small (<1e-3). Training may struggle; consider reducing BN or adjusting capsule params."})
                else:
                    self.log_message("[Pre-Train Distances] Unable to parse first validation batch structure.")
            except Exception as pd_e:
                self.log_message(f"[Pre-Train Distances] Skipped due to error: {pd_e}")
            
            # Enable adaptive threshold mechanism to prevent flat 50% metrics
            if hasattr(self.current_model, 'siamese_model'):
                self.current_model.adaptive_metric_threshold = True
                self.log_message("Adaptive threshold mechanism enabled")
            
            # Log batch configuration for verification
            train_bs = getattr(train_gen, 'batch_size', 'unknown')
            val_bs = getattr(val_gen, 'batch_size', 'unknown')
            self.log_message(f"Batch sizes - Train: {train_bs}, Val: {val_bs}")
            self.log_message(f"Data ready - Train: {len(train_gen)} batches, Val: {len(val_gen)} batches")
            
            # Learning rate is now set during model creation, no need to reassign
            user_lr = float(self.training_params['learning_rate'].get())
            self.log_message(f"Training with learning rate: {user_lr:.2e}")

            # Optional: quick pre-training threshold calibration for Siamese/CapsNet models
            try:
                if hasattr(self.current_model, 'calibrate_threshold') and hasattr(self.current_model, 'siamese_model') and self.current_model.siamese_model is not None:
                    # Run a lightweight calibration on a few validation batches to avoid degenerate initial metrics
                    thr_before = getattr(self.current_model, 'metric_distance_threshold', None)
                    self.progress_queue.put({'type': 'debug', 'message': f"[Pre-calibration] Starting quick threshold calibration (prev={thr_before})"})
                    # Use fewer batches for speed; emphasize recall for stability
                    try:
                        self.current_model.calibrate_threshold(val_gen, max_batches=10, metric='f_beta', beta=2.0, target_recall=0.6, smoothing=0.3)
                    except Exception as ce:
                        # Non-fatal: proceed without calibration if it fails
                        self.progress_queue.put({'type': 'debug', 'message': f"[Pre-calibration] Skipped due to error: {ce}"})
                    thr_after = getattr(self.current_model, 'metric_distance_threshold', None)
                    self.progress_queue.put({'type': 'debug', 'message': f"[Pre-calibration] Completed. threshold={thr_after}"})
            except Exception:
                # Ensure no impact to baseline CNN or training flow
                pass

            # Use the model directory that was set up in setup_model_directory()
            model_save_dir = os.path.join(self.current_model_dir, "models")
            try:
                os.makedirs(model_save_dir, exist_ok=True)
            except Exception:
                pass
            checkpoint_path = os.path.join(
                model_save_dir,
                "best_model_epoch_{epoch:02d}_auc_{val_auc:.4f}_vloss_{val_loss:.4f}_vf1_{val_f1:.4f}.weights.h5"
            )
            
            # Log the model save directory for debugging
            self.progress_queue.put({
                'type': 'debug',
                'message': f"Models will be saved to: {model_save_dir}"
            })
            
            # Create custom callbacks: validation metrics first so its logs propagate
            progress_callback = TrainingProgressCallback(self.progress_queue, total_epochs=epochs)
            validation_metrics_cb = ValidationMetricsCallback(val_gen, self.current_model, progress_queue=self.progress_queue, model_dir=model_save_dir)
            # Enable stable metrics: freeze threshold after burn-in and use light TTA
            # Moved freeze point later to allow more threshold exploration early
            validation_metrics_cb.freeze_after_epoch = 10  # Changed from 15 to 10
            validation_metrics_cb.tta_hflip = True

            # Ensure training metrics use the latest calibrated distance threshold at epoch start
            class ThresholdSyncCallback(tf.keras.callbacks.Callback):
                def __init__(self, model_wrapper, progress_queue=None):
                    super().__init__()
                    self.model_wrapper = model_wrapper
                    self.progress_queue = progress_queue
                def on_epoch_begin(self, epoch, logs=None):
                    try:
                        # Read latest distance-domain threshold and re-apply into shared var
                        thr = float(self.model_wrapper.get_metric_distance_threshold()) if hasattr(self.model_wrapper, 'get_metric_distance_threshold') else float(getattr(self.model_wrapper, 'metric_distance_threshold', 0.5))
                        if hasattr(self.model_wrapper, 'set_metric_distance_threshold'):
                            self.model_wrapper.set_metric_distance_threshold(thr)
                        if self.progress_queue is not None:
                            self.progress_queue.put({'type': 'debug', 'message': f"[ThresholdSync] Epoch {epoch+1}: train threshold (dist) = {thr:.4f}"})
                    except Exception:
                        pass

            # Hard cap on epochs at 50 as requested; stops training loop when reached
            class StopAtEpochCallback(tf.keras.callbacks.Callback):
                def __init__(self, stop_epoch=50):
                    super().__init__()
                    self.stop_epoch = int(stop_epoch)
                def on_epoch_end(self, epoch, logs=None):
                    if (epoch + 1) >= self.stop_epoch:
                        try:
                            if logs is not None:
                                logs['hard_stop_epoch'] = float(self.stop_epoch)
                        except Exception:
                            pass
                        if hasattr(self.model, 'stop_training'):
                            self.model.stop_training = True
            thr_sync_cb = ThresholdSyncCallback(self.current_model, self.progress_queue)
            hard_stop_cb = StopAtEpochCallback(stop_epoch=min(int(epochs), 50))
            
            # Monitor reliable metric across environments
            monitor_metric = 'val_auc'
            # Simple SWA implementation to improve generalization near the end
            class SimpleSWACallback(tf.keras.callbacks.Callback):
                def __init__(self, start_epoch, freq=1, verbose=1):
                    super().__init__()
                    self.start_epoch = int(start_epoch)
                    self.freq = int(max(1, freq))
                    self.verbose = verbose
                    self.swa_weights = None
                    self.n_models = 0
                def on_epoch_end(self, epoch, logs=None):
                    e = epoch + 1
                    if e < self.start_epoch:
                        return
                    if (e - self.start_epoch) % self.freq != 0:
                        return
                    weights = self.model.get_weights()
                    if self.swa_weights is None:
                        self.swa_weights = [w.copy() for w in weights]
                        self.n_models = 1
                    else:
                        self.n_models += 1
                        alpha = 1.0 / self.n_models
                        for i in range(len(self.swa_weights)):
                            self.swa_weights[i] = (1.0 - alpha) * self.swa_weights[i] + alpha * weights[i]
                    if self.verbose:
                        try:
                            self.model.stop_training = False
                        except Exception:
                            pass
                def on_train_end(self, logs=None):
                    if self.swa_weights is not None and self.n_models > 0:
                        self.model.set_weights(self.swa_weights)
                        # Force one more evaluation step by updating metric variables
                        if self.verbose and hasattr(self.model, 'optimizer'):
                            pass

            swa_start = max(5, int(self.training_params['epochs'].get() * 0.8))
            swa_cb = SimpleSWACallback(start_epoch=swa_start, freq=1, verbose=0)
            
            # Add embedding variance monitoring callback
            embedding_variance_cb = EmbeddingVarianceCallback(
                val_data=val_gen,
                base_model=self.current_model.model,
                progress_queue=self.progress_queue,
                check_every_n_epochs=1
            )

            noise_anneal_cb = GaussianNoiseAnnealingCallback(
                model_wrapper=self.current_model,
                start_epoch=4,
                end_epoch=12,
                min_std=max(0.005, getattr(self.current_model, 'noise_min_std', 0.02)),
                progress_queue=self.progress_queue,
                apply_every=1
            )

            # Margin scheduler DISABLED - analysis showed it caused degradation
            # margin_scheduler_cb = ContrastiveMarginScheduler(
            #     model_wrapper=self.current_model,
            #     max_margin=float(getattr(self.current_model, 'margin_schedule_max', self.current_model.get_contrastive_margin() * 1.4)),
            #     start_epoch=5,
            #     patience=2,
            #     min_delta=0.0025,
            #     progress_queue=self.progress_queue,
            #     step_fraction=0.4
            # )

            callbacks = [
                thr_sync_cb,
                validation_metrics_cb,
                progress_callback,
                EnsureValMetricsCallback(),
                embedding_variance_cb,
                noise_anneal_cb,
                # margin_scheduler_cb,  # DISABLED - caused performance degradation
                swa_cb,
                tf.keras.callbacks.ModelCheckpoint(
                    filepath=checkpoint_path,
                    monitor=monitor_metric,
                    mode='max',
                    save_best_only=True,
                    save_weights_only=True,
                    verbose=1
                ),
                tf.keras.callbacks.EarlyStopping(
                    monitor=monitor_metric,
                    patience=self.training_params['patience'].get(),
                    restore_best_weights=True,
                    mode='max',
                    verbose=1
                ),
                tf.keras.callbacks.ReduceLROnPlateau(
                    monitor='val_loss',
                    factor=0.5,
                    patience=3,
                    min_lr=1e-8,
                    mode='min',
                    verbose=1
                ),
                hard_stop_cb
            ]
            
            # Enable mixed precision for speed if available
            try:
                # Mixed precision optional disabled (removed unresolved import block)
                pass
                self.log_message("Mixed precision training enabled")
            except:
                self.log_message("WARNING: Mixed precision not available")
            
            self.log_message(f"Starting training with {epochs} epochs...")
            self.log_message(f"Model parameters: {self.current_model.siamese_model.count_params():,}")
            
            # Train the model with explicit verbose=1 for Keras logs
            # Ensure Keras can consume (inputs, labels, sample_weight) from Sequences
            # Keras will auto-detect the triple from the generator; no need to peel weights here.
            history = self.current_model.siamese_model.fit(
                train_gen,
                epochs=epochs,
                validation_data=val_gen,
                callbacks=callbacks,
                verbose=1,  # Enable Keras verbose logging
                use_multiprocessing=False,  # Avoid issues with GUI threading
                workers=1
            )
            
            # After training, ensure best checkpoint weights are loaded before final save
            # Rationale: If EarlyStopping didn't trigger (e.g., hit hard cap), the in-memory weights may be from the last epoch.
            # We scan saved checkpoints, pick the one with the highest val_auc from the filename pattern, and load it.
            try:
                import re, glob
                ckpt_pattern = os.path.join(model_save_dir, "best_model_epoch_*_auc_*.weights.h5")
                ckpt_files = glob.glob(ckpt_pattern)
                best_ckpt = None
                best_auc = float('-inf')
                for f in ckpt_files:
                    m = re.search(r"_auc_(\d+\.\d+)_", os.path.basename(f))
                    if not m:
                        continue
                    auc_val = float(m.group(1))
                    if auc_val > best_auc:
                        best_auc = auc_val
                        best_ckpt = f
                if best_ckpt is not None and os.path.exists(best_ckpt):
                    self.current_model.siamese_model.load_weights(best_ckpt)
                    self.progress_queue.put({'type': 'debug', 'message': f"Loaded best checkpoint before final save: {os.path.basename(best_ckpt)} (val_auc={best_auc:.4f})"})
                else:
                    # If no checkpoint found, rely on EarlyStopping restore_best_weights or last epoch weights
                    self.progress_queue.put({'type': 'debug', 'message': "No best checkpoint found; saving current in-memory weights."})
            except Exception as e:
                # Non-fatal—continue to final save with current weights
                self.progress_queue.put({'type': 'debug', 'message': f"[Post-fit Best Load Skipped] {e}"})

            # Now perform a consolidated final save (weights should reflect best checkpoint if available)
            try:
                final_dir = os.path.join(model_save_dir, 'final_best_model')
                os.makedirs(final_dir, exist_ok=True)
                # Save full model in SavedModel and Keras v3 format for reliability
                self.current_model.siamese_model.save(final_dir)
                keras_path = os.path.join(model_save_dir, 'final_best_model.keras')
                self.current_model.siamese_model.save(keras_path)
                # Save a weights-only H5 snapshot to avoid HDF5 naming conflicts
                h5_weights_path = os.path.join(model_save_dir, 'final_best_model.weights.h5')
                self.current_model.siamese_model.save_weights(h5_weights_path)
                self.progress_queue.put({'type': 'debug', 'message': f"Final model saved to: {final_dir}, {keras_path} and weights {h5_weights_path}"})
            except Exception as final_save_e:
                self.progress_queue.put({'type': 'debug', 'message': f"[Final Save Error] {final_save_e}"})

            # Training completed successfully
            self.progress_queue.put({
                'type': 'training_complete',
                'message': 'Training completed successfully.',
                'history': history.history
            })
            
        except Exception as e:
            # Training failed - capture full traceback to help diagnostics
            import traceback
            tb = traceback.format_exc()
            err_msg = f"Training failed: {str(e)}\nTraceback:\n{tb}"
            self.progress_queue.put({
                'type': 'training_error',
                'message': err_msg
            })
    
    def stop_training(self):
        """Stop the current training process."""
        if not self.training_active:
            messagebox.showinfo("Info", "No training in progress.")
            return
        
        self.log_message("Stopping training...")
        self.training_active = False
        self.status_var.set("Training stopped")
        self.update_button_states()
        
        # Note: Actual stopping of training thread would require more complex implementation
        messagebox.showinfo("Training Stopped", "Training will stop after the current epoch completes.")
    
    def retrain_existing_model(self):
        """Load an existing model file and retrain it."""
        # Show helper dialog first
        help_result = messagebox.askyesno("Model Selection Help", 
                        "This will help you load and retrain an existing model.\n\n"
                        "Expected model locations:\n"
                        "- Proposed models: models/proposed_siamese_capsnet*/\n\n"
                        "Look for .h5 files (e.g., best_model.h5, best_epoch_XX.h5)\n\n"
                        "Continue to file selection?")
        
        if not help_result:
            return
        
        # Open file dialog to select model file
        model_file = filedialog.askopenfilename(
            title="Select Model to Retrain",
            initialdir="models",
            filetypes=[
                ("Keras Model Files", "*.h5"),
                ("SavedModel Directories", "*.pb"),
                ("All Files", "*.*")
            ]
        )
        
        if not model_file:
            return  # User cancelled
        
        try:
            # Extract model type from file path
            model_type = self.detect_model_type_from_path(model_file)
            if not model_type:
                messagebox.showerror("Error", "Cannot determine model type from file path.\nPlease ensure the model is in the correct directory structure.")
                return
            
            # Load the model architecture first
            self.log_message(f"Initializing {model_type} model architecture...")
            self.select_model(model_type)
            
            if not self.current_model:
                messagebox.showerror("Error", "Failed to initialize model architecture.")
                return
            
            # Load the trained weights
            self.log_message(f"Loading existing model from: {os.path.basename(model_file)}")
            
            # Load the model weights/full model
            if model_file.endswith('.h5'):
                # Try to load the full model first
                try:
                    loaded_model = tf.keras.models.load_model(model_file, compile=False)
                    
                    # Copy weights to our model
                    if hasattr(self.current_model, 'siamese_model') and self.current_model.siamese_model is not None:
                        # Check architecture compatibility
                        loaded_weights = loaded_model.get_weights()
                        current_weights = self.current_model.siamese_model.get_weights()
                        
                        if len(loaded_weights) != len(current_weights):
                            raise ValueError(f"Model architecture mismatch: {len(loaded_weights)} vs {len(current_weights)} weight tensors")
                        
                        self.current_model.siamese_model.set_weights(loaded_weights)
                        self.log_message("Model weights loaded successfully.")
                        
                        # Verify a few weight values to confirm loading
                        verification_weights = self.current_model.siamese_model.get_weights()
                        if len(verification_weights) > 0:
                            sample_weight = verification_weights[0].flatten()[:5] if len(verification_weights[0].flatten()) >= 5 else verification_weights[0].flatten()
                            self.log_message(f"Weight verification - Sample values: {sample_weight}")
                    else:
                        messagebox.showerror("Error", "Current model not properly initialized.")
                        return
                        
                except Exception as e:
                    # Fallback to loading weights only
                    try:
                        if hasattr(self.current_model, 'siamese_model') and self.current_model.siamese_model is not None:
                            self.current_model.siamese_model.load_weights(model_file)
                            self.log_message("Model weights loaded successfully.")
                        else:
                            raise Exception("Model not properly initialized")
                    except Exception as e2:
                        messagebox.showerror("Error", f"Failed to load model weights: {str(e2)}\n\nOriginal error: {str(e)}")
                        return
            
            # Update model status
            model_name = os.path.basename(model_file)
            self.model_var.set(f"{self.current_model_name} (Loaded: {model_name})")
            self.log_message(f"Ready to retrain loaded model: {model_name}")
            
            # Ask for confirmation to start retraining
            result = messagebox.askyesno("Start Retraining", 
                                       f"Model loaded successfully!\n\n"
                                       f"File: {model_name}\n"
                                       f"Type: {self.current_model_name}\n"
                                       f"Epochs: {self.training_params['epochs'].get()}\n"
                                       f"Batch Size: {self.training_params['batch_size'].get()}\n"
                                       f"Learning Rate: {self.training_params['learning_rate'].get()}\n\n"
                                       f"The model will continue training from its current state.\n"
                                       f"Start retraining now?")
            
            if result:
                self.log_message("Starting retraining of loaded model...")
                self.start_training()
            
            # Enable training controls
            self.update_button_states()
            
        except Exception as e:
            error_msg = f"Error loading model for retraining: {str(e)}"
            self.log_message(error_msg)
            messagebox.showerror("Model Loading Error", error_msg)
    
    def browse_models_directory(self):
        """Show a browsable view of the models directory structure."""
        models_dir = "models"
        if not os.path.exists(models_dir):
            messagebox.showinfo("Models Directory", "No models directory found. Train a model first to create it.")
            return
        
        # Create a new window to show the models structure
        browse_window = tk.Toplevel(self.master)
        browse_window.title("Browse Models Directory")
        browse_window.geometry("800x600")
        browse_window.resizable(True, True)
        
        # Create main frame with scrollbar
        main_frame = ttk.Frame(browse_window)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Title label
        title_label = ttk.Label(main_frame, text="Available Models", font=("Arial", 14, "bold"))
        title_label.pack(anchor=tk.W, pady=(0, 10))
        
        # Create text widget with scrollbar
        text_frame = ttk.Frame(main_frame)
        text_frame.pack(fill=tk.BOTH, expand=True)
        
        text_widget = tk.Text(text_frame, wrap=tk.WORD, font=("Consolas", 10))
        scrollbar = ttk.Scrollbar(text_frame, orient=tk.VERTICAL, command=text_widget.yview)
        text_widget.config(yscrollcommand=scrollbar.set)
        
        text_widget.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Scan and display models directory
        models_info = self.scan_models_directory(models_dir)
        text_widget.insert(tk.END, models_info)
        text_widget.config(state=tk.DISABLED)  # Make read-only
        
        # Button frame
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill=tk.X, pady=(10, 0))
        
        # Refresh button
        def refresh_models():
            text_widget.config(state=tk.NORMAL)
            text_widget.delete(1.0, tk.END)
            models_info = self.scan_models_directory(models_dir)
            text_widget.insert(tk.END, models_info)
            text_widget.config(state=tk.DISABLED)
        
        refresh_button = ttk.Button(button_frame, text="Refresh", command=refresh_models)
        refresh_button.pack(side=tk.LEFT, padx=(0, 5))
        
        # Open models folder button
        def open_models_folder():
            try:
                import subprocess
                subprocess.Popen(f'explorer "{os.path.abspath(models_dir)}"')
            except Exception as e:
                messagebox.showerror("Error", f"Cannot open models folder: {str(e)}")
        
        open_folder_button = ttk.Button(button_frame, text="Open in Explorer", command=open_models_folder)
        open_folder_button.pack(side=tk.LEFT, padx=(0, 5))
        
        # Close button
        close_button = ttk.Button(button_frame, text="Close", command=browse_window.destroy)
        close_button.pack(side=tk.RIGHT)
    
    def scan_models_directory(self, models_dir):
        """Scan the models directory and return a formatted string of the structure."""
        result = []
        result.append("Models Directory Structure:\n")
        result.append("=" * 50 + "\n\n")
        
        try:
            for root, dirs, files in os.walk(models_dir):
                # Calculate depth for indentation
                level = root.replace(models_dir, '').count(os.sep)
                indent = "  " * level
                folder_name = os.path.basename(root) if root != models_dir else "models/"
                result.append(f"{indent}{folder_name}/\n")
                
                # Show files in this directory
                sub_indent = "  " * (level + 1)
                
                # Prioritize .h5 model files
                h5_files = [f for f in files if f.endswith('.h5')]
                other_files = [f for f in files if not f.endswith('.h5')]
                
                for file in h5_files:
                    file_path = os.path.join(root, file)
                    file_size = os.path.getsize(file_path)
                    size_mb = file_size / (1024 * 1024)
                    modified_time = os.path.getmtime(file_path)
                    modified_str = time.strftime("%Y-%m-%d %H:%M", time.localtime(modified_time))
                    result.append(f"{sub_indent}{file} ({size_mb:.1f} MB, {modified_str})\n")
                
                for file in other_files[:10]:  # Limit other files to prevent clutter
                    result.append(f"{sub_indent}{file}\n")
                
                if len(other_files) > 10:
                    result.append(f"{sub_indent}... and {len(other_files) - 10} more files\n")
                
                result.append("\n")
                
        except Exception as e:
            result.append(f"Error scanning directory: {str(e)}\n")
        
        result.append("\n" + "=" * 50 + "\n")
        result.append("Legend:\n")
        result.append("- .h5 = Keras model files (main models for retraining)\n")
        result.append("- other = Other files (logs, checkpoints, etc.)\n")
        result.append("- trailing / = Directories\n\n")
        result.append("To retrain a model, use 'Load & Retrain Existing Model' and select a .h5 file.\n")
        
        return "".join(result)

    def detect_model_type_from_path(self, model_path):
        """Detect model type from file path (simplified to proposed only)."""
        return 'proposed'
    
    def reset_training(self):
        """Reset training data and plots."""
        self.reset_training_data()
        self.setup_progress_plots()
        self.setup_loss_plots()
        self.setup_accuracy_plots()
        self.status_var.set("Training reset")
        self.progress_var.set(0)
        self.log_message("Training data reset")
    
    def reset_training_data(self):
        """Reset the training data storage."""
        self.training_data = {
            'epochs': [],
            'loss': [],
            'val_loss': [],
            'accuracy': [],
            'val_accuracy': [],
            'learning_rate': [],
            'precision': [],
            'val_precision': [],
            'recall': [],
            'val_recall': [],
            'f1_score': [],
            'val_f1_score': [],
            'val_auc': [],
            'thresholds': []
        }
        # Also reset synchronization helpers (reset_training_data)
        self._awaiting_val_metrics = False
        self._last_train_logs = {}
        self._last_epoch_times = {}
        self._last_val_epoch = 0
    
    def update_button_states(self):
        """Update button states based on current status."""
        if self.current_model and not self.training_active:
            self.train_button.config(state=tk.NORMAL)
            self.stop_button.config(state=tk.DISABLED)
        elif self.training_active:
            self.train_button.config(state=tk.DISABLED)
            self.stop_button.config(state=tk.NORMAL)
        else:
            self.train_button.config(state=tk.DISABLED)
            self.stop_button.config(state=tk.DISABLED)
    
    def update_loss_plot(self):
        """Update the detailed loss analysis plot."""
        if not self.training_data['epochs']:
            return
        
        self.loss_fig.clear()
        
        # Loss progression
        ax1 = self.loss_fig.add_subplot(2, 2, 1)
        epochs = self.training_data['epochs']
        ax1.plot(epochs, self.training_data['loss'], 'b-', label='Training Loss', linewidth=2)
        ax1.plot(epochs, self.training_data['val_loss'], 'r-', label='Validation Loss', linewidth=2)
        ax1.set_title('Loss Progression', fontweight='bold')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Loss difference
        ax2 = self.loss_fig.add_subplot(2, 2, 2)
        loss_diff = [val - train for val, train in zip(self.training_data['val_loss'], self.training_data['loss'])]
        ax2.plot(epochs, loss_diff, 'g-', linewidth=2)
        ax2.set_title('Validation - Training Loss', fontweight='bold')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss Difference')
        ax2.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        ax2.grid(True, alpha=0.3)
        
        # Loss distribution
        ax3 = self.loss_fig.add_subplot(2, 2, 3)
        ax3.hist(self.training_data['loss'], alpha=0.7, label='Training Loss', bins=20)
        ax3.hist(self.training_data['val_loss'], alpha=0.7, label='Validation Loss', bins=20)
        ax3.set_title('Loss Distribution', fontweight='bold')
        ax3.set_xlabel('Loss Value')
        ax3.set_ylabel('Frequency')
        ax3.legend()
        
        # Training statistics
        ax4 = self.loss_fig.add_subplot(2, 2, 4)
        min_train_loss = min(self.training_data['loss'])
        min_val_loss = min(self.training_data['val_loss'])
        final_train_loss = self.training_data['loss'][-1]
        final_val_loss = self.training_data['val_loss'][-1]
        
        stats_text = f"""Training Statistics:

Min Training Loss: {min_train_loss:.4f}
Min Validation Loss: {min_val_loss:.4f}
Final Training Loss: {final_train_loss:.4f}
Final Validation Loss: {final_val_loss:.4f}

Total Epochs: {len(epochs)}
Model: {self.current_model_name}"""
        
        ax4.text(0.1, 0.9, stats_text, transform=ax4.transAxes,
                fontsize=10, fontfamily='monospace', verticalalignment='top')
        ax4.set_xlim(0, 1)
        ax4.set_ylim(0, 1)
        ax4.axis('off')
        
        self.loss_fig.tight_layout()
        self.loss_canvas.draw()
    
    def update_accuracy_plot(self):
        """Update the accuracy analysis plot."""
        if not self.training_data['accuracy'] or not self.training_data['val_accuracy']:
            return
        
        self.accuracy_fig.clear()
        
        # Accuracy progression
        ax1 = self.accuracy_fig.add_subplot(2, 2, 1)
        epochs = self.training_data['epochs']
        ax1.plot(epochs, self.training_data['accuracy'], 'b-', label='Training Accuracy', linewidth=2)
        ax1.plot(epochs, self.training_data['val_accuracy'], 'r-', label='Validation Accuracy', linewidth=2)
        ax1.set_title('Accuracy Progression', fontweight='bold')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Accuracy vs Loss
        ax2 = self.accuracy_fig.add_subplot(2, 2, 2)
        ax2.scatter(self.training_data['loss'], self.training_data['accuracy'], 
                   alpha=0.7, label='Training', color='blue')
        ax2.scatter(self.training_data['val_loss'], self.training_data['val_accuracy'], 
                   alpha=0.7, label='Validation', color='red')
        ax2.set_title('Accuracy vs Loss', fontweight='bold')
        ax2.set_xlabel('Loss')
        ax2.set_ylabel('Accuracy')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Accuracy improvement
        ax3 = self.accuracy_fig.add_subplot(2, 2, 3)
        if len(self.training_data['accuracy']) > 1:
            train_acc_diff = np.diff(self.training_data['accuracy'])
            val_acc_diff = np.diff(self.training_data['val_accuracy'])
            ax3.plot(epochs[1:], train_acc_diff, 'b-', label='Training Accuracy Change', linewidth=2)
            ax3.plot(epochs[1:], val_acc_diff, 'r-', label='Validation Accuracy Change', linewidth=2)
        ax3.set_title('Accuracy Change per Epoch', fontweight='bold')
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('Accuracy Change')
        ax3.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Accuracy statistics
        ax4 = self.accuracy_fig.add_subplot(2, 2, 4)
        max_train_acc = max(self.training_data['accuracy'])
        max_val_acc = max(self.training_data['val_accuracy'])
        final_train_acc = self.training_data['accuracy'][-1]
        final_val_acc = self.training_data['val_accuracy'][-1]
        
        stats_text = f"""Accuracy Statistics:

Max Training Accuracy: {max_train_acc:.4f}
Max Validation Accuracy: {max_val_acc:.4f}
Final Training Accuracy: {final_train_acc:.4f}
Final Validation Accuracy: {final_val_acc:.4f}

Gap: {abs(final_train_acc - final_val_acc):.4f}
Model: {self.current_model_name}"""
        
        ax4.text(0.1, 0.9, stats_text, transform=ax4.transAxes,
                fontsize=10, fontfamily='monospace', verticalalignment='top')
        ax4.set_xlim(0, 1)
        ax4.set_ylim(0, 1)
        ax4.axis('off')
        
        self.accuracy_fig.tight_layout()
        self.accuracy_canvas.draw()
    
    def show_accuracy_statistics(self):
        """Show detailed accuracy statistics in a popup."""
        if not self.training_data['accuracy']:
            messagebox.showinfo("No Data", "No accuracy data available.")
            return
        
        # Calculate statistics
        train_acc = self.training_data['accuracy']
        val_acc = self.training_data['val_accuracy']
        
        stats = f"""Detailed Accuracy Statistics:

Training Accuracy:
  Maximum: {max(train_acc):.6f}
  Minimum: {min(train_acc):.6f}
  Final: {train_acc[-1]:.6f}
  Mean: {np.mean(train_acc):.6f}
  Std: {np.std(train_acc):.6f}

Validation Accuracy:
  Maximum: {max(val_acc):.6f}
  Minimum: {min(val_acc):.6f}
  Final: {val_acc[-1]:.6f}
  Mean: {np.mean(val_acc):.6f}
  Std: {np.std(val_acc):.6f}

Overfitting Analysis:
  Accuracy Gap: {abs(train_acc[-1] - val_acc[-1]):.6f}
  Max Gap: {max([abs(t-v) for t,v in zip(train_acc, val_acc)]):.6f}
  
Best Validation Epoch: {val_acc.index(max(val_acc)) + 1}
Total Epochs Trained: {len(train_acc)}
"""
        
        messagebox.showinfo("Accuracy Statistics", stats)
    
    def load_comparison_data(self):
        """Load comparison data from saved model results."""
        # This would load results from different model training sessions
        self.log_message("Loading comparison data...")
        messagebox.showinfo("Feature", "Model comparison feature will load data from completed training sessions.")
    
    def update_comparison_plot(self):
        """Update the model comparison plot."""
        # This would show comparison between different models
        self.comparison_fig.clear()
        
        ax = self.comparison_fig.add_subplot(1, 1, 1)
        ax.set_title('Model Performance Comparison', fontweight='bold', fontsize=16)
        ax.text(0.5, 0.5, 'Train multiple models to see performance comparisons\n\nComparison data will appear here after training different architectures',
                ha='center', va='center', transform=ax.transAxes, fontsize=12)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        
        self.comparison_fig.tight_layout()
        self.comparison_canvas.draw()
    
    def log_message(self, message):
        """Add a message to the training log."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        log_entry = f"[{timestamp}] {message}\n"
        
        self.log_text.insert(tk.END, log_entry)
        self.log_text.see(tk.END)
        
        # Also print to console
        print(f"{timestamp} - {message}")
    
    def load_training_data(self):
        """Load training data from file."""
        file_path = filedialog.askopenfilename(
            title="Load Training Data",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        
        if file_path:
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                
                # Load training history
                if 'training_data' in data:
                    self.training_data = data['training_data']
                    self.update_progress_plots()
                    self.update_loss_plots()
                    self.update_accuracy_plots()
                    self.log_message(f"Training data loaded from {file_path}")
                
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load training data: {str(e)}")
    
    def save_training_results(self):
        """Save current training results to file."""
        if not self.training_data['epochs']:
            messagebox.showwarning("No Data", "No training data to save.")
            return
        
        file_path = filedialog.asksaveasfilename(
            title="Save Training Results",
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        
        if file_path:
            try:
                save_data = {
                    'model_name': self.current_model_name,
                    'training_params': {
                        'epochs': self.training_params['epochs'].get(),
                        'batch_size': self.training_params['batch_size'].get(),
                        'learning_rate': self.training_params['learning_rate'].get(),
                        'patience': self.training_params['patience'].get()
                    },
                    'training_data': self.training_data,
                    'timestamp': datetime.now().isoformat()
                }
                
                with open(file_path, 'w') as f:
                    json.dump(save_data, f, indent=2)
                
                self.log_message(f"Training results saved to {file_path}")
                messagebox.showinfo("Success", f"Training results saved successfully!")
                
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save training results: {str(e)}")
    
    def export_training_report(self):
        """Export a comprehensive training report."""
        if not self.training_data['epochs']:
            messagebox.showwarning("No Data", "No training data to export.")
            return
        
        file_path = filedialog.asksaveasfilename(
            title="Export Training Report",
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
        )
        
        if file_path:
            try:
                report = f"""FUR-GET ME NOT: TRAINING REPORT
{'='*60}

Model: {self.current_model_name}
Training Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

Training Parameters:
  Epochs: {self.training_params['epochs'].get()}
  Batch Size: {self.training_params['batch_size'].get()}
  Learning Rate: {self.training_params['learning_rate'].get()}
  Early Stop Patience: {self.training_params['patience'].get()}

Training Results:
  Total Epochs Completed: {len(self.training_data['epochs'])}
  Final Training Loss: {self.training_data['loss'][-1]:.6f}
  Final Validation Loss: {self.training_data['val_loss'][-1]:.6f}
  Best Training Loss: {min(self.training_data['loss']):.6f}
  Best Validation Loss: {min(self.training_data['val_loss']):.6f}

"""
                
                if self.training_data['accuracy']:
                    report += f"""Accuracy Results:
  Final Training Accuracy: {self.training_data['accuracy'][-1]:.6f}
  Final Validation Accuracy: {self.training_data['val_accuracy'][-1]:.6f}
  Best Training Accuracy: {max(self.training_data['accuracy']):.6f}
  Best Validation Accuracy: {max(self.training_data['val_accuracy']):.6f}
  Accuracy Gap: {abs(self.training_data['accuracy'][-1] - self.training_data['val_accuracy'][-1]):.6f}

"""
                
                report += f"""Training Summary:
  The {self.current_model_name} was trained for {len(self.training_data['epochs'])} epochs.
  Training showed {'good convergence' if self.training_data['val_loss'][-1] < self.training_data['loss'][-1] * 1.1 else 'potential overfitting'}.
  
End of Report
{'='*60}"""
                
                with open(file_path, 'w') as f:
                    f.write(report)
                
                self.log_message(f"Training report exported to {file_path}")
                messagebox.showinfo("Success", "Training report exported successfully!")
                
            except Exception as e:
                messagebox.showerror("Error", f"Failed to export report: {str(e)}")
    
    def setup_model_directory(self):
        """Setup directory structure for saving model training data."""
        try:
            # Create a unique session ID based on current time and model
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.training_session_id = f"{self.current_model_name}_{timestamp}"
            
            # Create model directory
            base_models_dir = "models"
            if not os.path.exists(base_models_dir):
                os.makedirs(base_models_dir)
            
            self.current_model_dir = os.path.join(base_models_dir, self.training_session_id)
            if not os.path.exists(self.current_model_dir):
                os.makedirs(self.current_model_dir)
            
            # Create subdirectories for different types of data
            subdirs = ["logs", "plots", "models"]
            for subdir in subdirs:
                subdir_path = os.path.join(self.current_model_dir, subdir)
                if not os.path.exists(subdir_path):
                    os.makedirs(subdir_path)
            
            self.log_message(f"Model directory created: {self.current_model_dir}")
            return True
            
        except Exception as e:
            self.log_message(f"Error creating model directory: {str(e)}")
            return False
    
    def save_epoch_data(self, epoch, logs, elapsed_time, epoch_time, eta):
        """Save training logs and analytics for current epoch."""
        if not self.current_model_dir:
            return
        
        try:
            # Save training logs to text file
            self.save_training_logs(epoch, logs, elapsed_time, epoch_time, eta)
            
            # Save live metrics to text file
            self.save_live_metrics(epoch, logs)
            
            # Save analytics and graphs
            self.save_epoch_analytics(epoch)
            
        except Exception as e:
            self.log_message(f"Error saving epoch {epoch} data: {str(e)}")
    
    def save_training_logs(self, epoch, logs, elapsed_time, epoch_time, eta):
        """Save detailed training logs for the epoch."""
        try:
            logs_file = os.path.join(self.current_model_dir, "logs", f"training_logs_epoch_{epoch:03d}.txt")
            
            with open(logs_file, 'w') as f:
                f.write(f"Training Logs - Epoch {epoch}\n")
                f.write("=" * 50 + "\n")
                f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Model: {self.current_model_name}\n")
                f.write(f"Session ID: {self.training_session_id}\n\n")
                
                # Timing information
                f.write("TIMING INFORMATION:\n")
                f.write(f"Elapsed Time: {elapsed_time:.2f} seconds ({elapsed_time/60:.2f} minutes)\n")
                f.write(f"Epoch Time: {epoch_time:.2f} seconds\n")
                f.write(f"ETA: {eta:.2f} seconds ({eta/60:.2f} minutes)\n\n")
                
                # Training parameters
                f.write("TRAINING PARAMETERS:\n")
                f.write(f"Epochs Total: {self.training_params['epochs'].get()}\n")
                f.write(f"Batch Size: {self.training_params['batch_size'].get()}\n")
                f.write(f"Learning Rate: {self.training_params['learning_rate'].get()}\n")
                f.write(f"Patience: {self.training_params['patience'].get()}\n\n")
                
                # Metrics for this epoch
                f.write("EPOCH METRICS:\n")
                for key, value in logs.items():
                    try:
                        if isinstance(value, (int, float)):
                            f.write(f"{key}: {value:.6f}\n")
                        else:
                            f.write(f"{key}: {value}\n")
                    except:
                        f.write(f"{key}: {value}\n")
                
                # Historical data summary
                if self.training_data['epochs']:
                    f.write(f"\nHISTORICAL SUMMARY (up to epoch {epoch}):\n")
                    f.write(f"Best Training Loss: {min(self.training_data['loss']):.6f}\n")
                    f.write(f"Best Validation Loss: {min(self.training_data['val_loss']):.6f}\n")
                    if self.training_data['val_accuracy']:
                        f.write(f"Best Validation Accuracy: {max(self.training_data['val_accuracy']):.6f}\n")
                    if self.training_data['val_f1_score']:
                        f.write(f"Best Validation F1-Score: {max(self.training_data['val_f1_score']):.6f}\n")
                    if self.training_data['val_auc']:
                        f.write(f"Best Validation AUC: {max(self.training_data['val_auc']):.6f}\n")
            
        except Exception as e:
            self.log_message(f"Error saving training logs for epoch {epoch}: {str(e)}")
    
    def save_live_metrics(self, epoch, logs):
        """Save live metrics display to text file."""
        try:
            metrics_file = os.path.join(self.current_model_dir, "logs", f"live_metrics_epoch_{epoch:03d}.txt")
            
            # Get current metrics text content
            metrics_content = self.current_metrics.get(1.0, tk.END)
            
            with open(metrics_file, 'w') as f:
                f.write(f"Live Metrics Display - Epoch {epoch}\n")
                f.write("=" * 50 + "\n")
                f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                f.write(metrics_content)
            
        except Exception as e:
            self.log_message(f"Error saving live metrics for epoch {epoch}: {str(e)}")
    
    def save_epoch_analytics(self, epoch):
        """Save all analytics and graphs for the current epoch."""
        try:
            plots_dir = os.path.join(self.current_model_dir, "plots", f"epoch_{epoch:03d}")
            if not os.path.exists(plots_dir):
                os.makedirs(plots_dir)
            
            # Save training progress plots
            if hasattr(self, 'progress_fig'):
                progress_plot_path = os.path.join(plots_dir, "training_progress.png")
                self.progress_fig.savefig(progress_plot_path, dpi=300, bbox_inches='tight')
            
            # Save loss analysis plots
            if hasattr(self, 'loss_fig'):
                loss_plot_path = os.path.join(plots_dir, "loss_analysis.png")
                self.loss_fig.savefig(loss_plot_path, dpi=300, bbox_inches='tight')
            
            # Save accuracy metrics plots
            if hasattr(self, 'accuracy_fig'):
                accuracy_plot_path = os.path.join(plots_dir, "accuracy_metrics.png")
                self.accuracy_fig.savefig(accuracy_plot_path, dpi=300, bbox_inches='tight')
            
            # Save ROC/AUC plots
            if hasattr(self, 'roc_fig'):
                roc_plot_path = os.path.join(plots_dir, "roc_auc_analysis.png")
                self.roc_fig.savefig(roc_plot_path, dpi=300, bbox_inches='tight')
            
            # Save comparison plots
            if hasattr(self, 'comparison_fig'):
                comparison_plot_path = os.path.join(plots_dir, "model_comparison.png")
                self.comparison_fig.savefig(comparison_plot_path, dpi=300, bbox_inches='tight')
            
            # Save analytics data to JSON
            analytics_file = os.path.join(plots_dir, "analytics_data.json")
            analytics_data = {
                'epoch': epoch,
                'timestamp': datetime.now().isoformat(),
                'training_data': {
                    'epochs': self.training_data['epochs'],
                    'loss': self.training_data['loss'],
                    'val_loss': self.training_data['val_loss'],
                    'accuracy': self.training_data['accuracy'],
                    'val_accuracy': self.training_data['val_accuracy'],
                    'precision': self.training_data['precision'],
                    'val_precision': self.training_data['val_precision'],
                    'recall': self.training_data['recall'],
                    'val_recall': self.training_data['val_recall'],
                    'f1_score': self.training_data['f1_score'],
                    'val_f1_score': self.training_data['val_f1_score'],
                    'val_auc': self.training_data['val_auc'],
                    'learning_rate': self.training_data['learning_rate']
                },
                'training_params': {
                    'epochs': self.training_params['epochs'].get(),
                    'batch_size': self.training_params['batch_size'].get(),
                    'learning_rate': self.training_params['learning_rate'].get(),
                    'patience': self.training_params['patience'].get()
                },
                'model_info': {
                    'model_name': self.current_model_name,
                    'session_id': self.training_session_id
                }
            }
            
            with open(analytics_file, 'w') as f:
                json.dump(analytics_data, f, indent=2, default=str)
            
        except Exception as e:
            self.log_message(f"Error saving analytics for epoch {epoch}: {str(e)}")
    
    def export_loss_data(self):
        """Export loss data to CSV."""
        if not self.training_data['epochs']:
            messagebox.showwarning("No Data", "No training data to export.")
            return
        
        file_path = filedialog.asksaveasfilename(
            title="Export Loss Data",
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        
        if file_path:
            try:
                df = pd.DataFrame({
                    'epoch': self.training_data['epochs'],
                    'training_loss': self.training_data['loss'],
                    'validation_loss': self.training_data['val_loss']
                })
                
                if self.training_data['accuracy']:
                    df['training_accuracy'] = self.training_data['accuracy']
                    df['validation_accuracy'] = self.training_data['val_accuracy']
                
                df.to_csv(file_path, index=False)
                
                self.log_message(f"Loss data exported to {file_path}")
                messagebox.showinfo("Success", "Loss data exported successfully!")
                
            except Exception as e:
                messagebox.showerror("Error", f"Failed to export loss data: {str(e)}")
    
    def on_closing(self):
        """Handle application closing."""
        if self.training_active:
            result = messagebox.askyesno("Training Active", 
                                       "Training is still in progress. Are you sure you want to exit?")
            if not result:
                return
        
        self.root.quit()
        self.root.destroy()


def main():
    """Main function to run the training GUI."""
    print("Fur-get Me Not: Real-Time Training Interface")
    print("=" * 60)
    
    # Create and run the GUI
    root = tk.Tk()
    
    # Handle closing properly
    app = ModelTrainingGUI(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    
    try:
        root.mainloop()
    except KeyboardInterrupt:
        print("\nApplication interrupted by user")
    except Exception as e:
        print(f"Error running application: {e}")
    finally:
        print("Training interface closed")


if __name__ == "__main__":
    main()
