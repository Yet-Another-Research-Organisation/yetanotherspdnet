"""Useful utilities for metrics"""

import json
import time
from pathlib import Path
from typing import Dict, Any, Optional
import psutil
import torch

try:
    from pyJoules.device.rapl_device import RaplPackageDomain, RaplDramDomain, RaplDevice
    from pyJoules.device.nvidia_device import NvidiaGPUDomain, NvidiaGPUDevice
    PYJOULE_AVAILABLE = True
except ImportError:
    PYJOULE_AVAILABLE = False
    print("Warning: pyJoule not available. Energy monitoring will be disabled.")

import threading


class MetricsWriter:
    def __init__(
        self, log_dir: str = "training_logs", experiment_name: str = "experiment"
    ):
        """
        Initialize metrics writer

        Args:
            log_dir: Directory to store metrics files
            experiment_name: Name of the experiment
        """
        self.log_dir = Path(log_dir)
        self.experiment_name = experiment_name
        self.log_dir.mkdir(exist_ok=True)

        # Create metrics file
        self.metrics_file = self.log_dir / f"{experiment_name}_metrics.json"
        self.status_file = self.log_dir / f"{experiment_name}_status.json"
        
        # Track process
        self.process = psutil.Process()
        self.start_ram_usage = self.process.memory_info().rss / (1024**3)  # GB
        
        # Energy and RAM monitoring
        self.energy_monitoring_active = False
        self.monitoring_thread = None
        self.energy_samples = []
        self.ram_samples = []
        self.monitoring_interval = 1.0  # seconds
        
        # pyJoules setup
        self.pyjoule_available = PYJOULE_AVAILABLE
        self.energy_domains = []
        self.energy_devices = []
        
        if PYJOULE_AVAILABLE:
            try:
                from pyJoules.device.rapl_device import RaplDevice
                from pyJoules.device.nvidia_device import NvidiaGPUDevice
                
                # Setup RAPL domains - try each and keep only available ones
                rapl_domains = []
                
                # Always try package domain (CPU total)
                try:
                    domain = RaplPackageDomain(0)
                    test_device = RaplDevice()
                    test_device.configure(domains=[domain])
                    rapl_domains.append(domain)
                except:
                    pass
                
                # Try DRAM domain (may not exist on laptops)
                try:
                    domain = RaplDramDomain(0)
                    test_device = RaplDevice()
                    test_device.configure(domains=[domain])
                    rapl_domains.append(domain)
                except:
                    pass
                
                # Create and configure RAPL device with available domains
                if rapl_domains:
                    rapl_device = RaplDevice()
                    rapl_device.configure(domains=rapl_domains)
                    self.energy_devices.append(rapl_device)
                    self.energy_domains.extend(rapl_domains)
                
                # Try to add GPU if available
                if torch.cuda.is_available():
                    try:
                        gpu_domain = NvidiaGPUDomain(0)
                        gpu_device = NvidiaGPUDevice()
                        gpu_device.configure(domains=[gpu_domain])
                        self.energy_devices.append(gpu_device)
                        self.energy_domains.append(gpu_domain)
                        print("✓ GPU energy monitoring enabled")
                    except Exception as e:
                        print(f"Note: GPU energy monitoring not available: {e}")
                
                print(f"✓ pyJoules initialized with {len(self.energy_devices)} device(s), {len(self.energy_domains)} domain(s)")
            except Exception as e:
                print(f"Warning: Could not initialize pyJoules: {e}")
                import traceback
                traceback.print_exc()
                self.pyjoule_available = False

        # Initialize status
        self._write_status(
            {
                "experiment_name": experiment_name,
                "status": "starting",
                "start_time": time.time(),
                "current_epoch": 0,
                "total_epochs": 0,
                "best_val_acc": 0.0,
                "best_val_loss": float("inf"),
            }
        )

    def log_metrics(self, epoch: int, step: int, metrics: Dict[str, Any]):
        """Log training metrics"""
        entry = {"timestamp": time.time(), "epoch": epoch, "step": step, **metrics}

        with open(self.metrics_file, "a") as f:
            f.write(json.dumps(entry) + "\n")

    def log_model_info(
        self, model_name: str, total_params: Optional[int] = None, **kwargs
    ):
        """Log model information"""
        model_info = {"model_name": model_name, "total_params": total_params, **kwargs}
        self.update_status({"model_info": model_info})

    def update_status(self, status_update: Dict[str, Any]):
        """Update training status"""
        # Read current status
        if self.status_file.exists():
            with open(self.status_file, "r") as f:
                current_status = json.load(f)
        else:
            current_status = {}

        # Update with new values
        current_status.update(status_update)
        current_status["last_update"] = time.time()

        self._write_status(current_status)

    def _write_status(self, status: Dict[str, Any]):
        """Write status to file"""
        with open(self.status_file, "w") as f:
            json.dump(status, f, indent=2)

    def _monitoring_loop(self):
        """Background thread for continuous energy and RAM monitoring"""
        if not self.pyjoule_available:
            return
        
        try:
            from pyJoules.energy_meter import EnergyMeter
            
            # Create ONE meter for all measurements (reused across iterations)
            meter = EnergyMeter(self.energy_devices)
            
            while self.energy_monitoring_active:
                try:
                    # Start energy measurement
                    meter.start()
                    
                    # Wait for monitoring interval
                    time.sleep(self.monitoring_interval)
                    
                    # Stop measurement
                    meter.stop()
                except (PermissionError, OSError) as e:
                    if 'Permission denied' in str(e) or e.errno == 13:
                        print(f"\nWarning: Cannot access RAPL energy counters (Permission denied).")
                        print(f"Energy monitoring disabled. Run with sudo or configure permissions:")
                        print(f"  sudo chmod -R a+r /sys/class/powercap/intel-rapl")
                        self.energy_monitoring_active = False
                        self.pyjoule_available = False
                        break
                    else:
                        print(f"Error reading energy: {e}")
                        time.sleep(self.monitoring_interval)
                        continue
                except Exception as e:
                    print(f"Error reading energy: {e}")
                    time.sleep(self.monitoring_interval)
                    continue
                
                end_time = time.time()
                
                # Get trace - accumulates all start/stop cycles
                trace = meter.get_trace()
                if trace and len(trace) > 0:
                    sample = trace[-1]  # Get most recent measurement
                    energy_data = {
                        'timestamp': end_time,
                        'duration': sample.duration / 1e6,  # µs to s
                    }
                    
                    # Extract energy values
                    for tag, energy_raw in sample.energy.items():
                        # Clean up device name
                        device_name = str(tag).lower()
                        # Remove suffixes and simplify
                        for suffix in ['_0', 'domain', 'rapl']:
                            device_name = device_name.replace(suffix, '')
                        device_name = device_name.strip('_')
                        
                        # Rename for clarity and convert to Joules based on device type
                        if 'package' in device_name:
                            device_name = 'cpu'
                            # RAPL returns microjoules (µJ)
                            energy_joules = energy_raw / 1e6
                        elif 'nvidia' in device_name or device_name == 'gpu':
                            device_name = 'gpu'
                            # NVIDIA GPU returns millijoules (mJ)
                            energy_joules = energy_raw / 1e3
                        elif 'dram' in device_name:
                            device_name = 'ram'
                            # RAPL returns microjoules (µJ)
                            energy_joules = energy_raw / 1e6
                        else:
                            # Default: assume microjoules
                            energy_joules = energy_raw / 1e6
                        
                        # Store energy in Joules
                        energy_data[f'{device_name}_joules'] = energy_joules
                    
                    self.energy_samples.append(energy_data)
                
                # Record RAM usage
                ram_gb = self.process.memory_info().rss / (1024**3)
                self.ram_samples.append({
                    'timestamp': end_time,
                    'ram_gb': ram_gb
                })
                
        except Exception as e:
            print(f"Error in energy monitoring loop: {e}")
            import traceback
            traceback.print_exc()
            self.energy_monitoring_active = False
    
    def start_energy_monitoring(self):
        """Start continuous energy and RAM monitoring in background thread"""
        if not self.pyjoule_available:
            print("Warning: pyJoules not available, skipping energy monitoring")
            return
        
        if not self.energy_domains:
            print("Warning: No energy domains configured")
            return
        
        if self.energy_monitoring_active:
            print("Warning: Energy monitoring already active")
            return
        
        self.energy_monitoring_active = True
        self.energy_samples = []
        self.ram_samples = []
        
        # Start monitoring thread
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitoring_thread.start()
        
        print(f"✓ Energy monitoring started (sampling every {self.monitoring_interval}s)")
    
    def stop_energy_monitoring(self):
        """Stop energy monitoring and compute statistics"""
        if not self.energy_monitoring_active:
            return {}
        
        self.energy_monitoring_active = False
        
        # Wait for thread to finish
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5.0)
        
        # Compute statistics
        stats = {
            'samples_count': len(self.energy_samples),
            'monitoring_duration_seconds': 0,
        }
        
        if self.energy_samples:
            # Total monitoring duration
            stats['monitoring_duration_seconds'] = (
                self.energy_samples[-1]['timestamp'] - self.energy_samples[0]['timestamp']
            )
            
            # Aggregate energy by component
            components = set()
            for sample in self.energy_samples:
                for key in sample.keys():
                    if key.endswith('_joules'):
                        component = key.replace('_joules', '')
                        components.add(component)
            
            # Calculate totals and averages for each component
            total_energy = 0
            for component in components:
                joules_key = f'{component}_joules'
                watts_key = f'{component}_watts'
                
                # Sum all joules for this component
                total_joules = sum(s.get(joules_key, 0) for s in self.energy_samples)
                stats[f'{component}_total_joules'] = total_joules
                
                # Average power
                if stats['monitoring_duration_seconds'] > 0:
                    stats[f'{component}_avg_watts'] = total_joules / stats['monitoring_duration_seconds']
                else:
                    stats[f'{component}_avg_watts'] = 0
                
                # Max instantaneous power
                watts_values = [s.get(watts_key, 0) for s in self.energy_samples if watts_key in s]
                if watts_values:
                    stats[f'{component}_max_watts'] = max(watts_values)
                    stats[f'{component}_min_watts'] = min(watts_values)
                
                total_energy += total_joules
            
            # Total energy and average power across all components
            stats['total_energy_joules'] = total_energy
            stats['total_energy_kwh'] = total_energy / 3.6e6  # Convert J to kWh
            if stats['monitoring_duration_seconds'] > 0:
                stats['total_avg_watts'] = total_energy / stats['monitoring_duration_seconds']
        
        # RAM statistics
        if self.ram_samples:
            ram_values = [s['ram_gb'] for s in self.ram_samples]
            stats['ram_avg_gb'] = sum(ram_values) / len(ram_values)
            stats['ram_max_gb'] = max(ram_values)
            stats['ram_min_gb'] = min(ram_values)
            stats['ram_samples_count'] = len(ram_values)
        
        print(f"✓ Energy monitoring stopped ({len(self.energy_samples)} energy samples, {len(self.ram_samples)} RAM samples)")
        if self.energy_samples:
            print(f"  Total energy: {stats['total_energy_joules']:.2f} J ({stats['total_energy_kwh']:.6f} kWh)")
            print(f"  Average power: {stats.get('total_avg_watts', 0):.2f} W")
        
        return stats

    def start_training(self, total_epochs: int):
        """Start training phase and begin energy monitoring"""
        # Start continuous monitoring
        self.start_energy_monitoring()
        
        self.update_status(
            {
                "status": "training", 
                "total_epochs": total_epochs, 
                "current_epoch": 0,
            }
        )

    def log_epoch_metrics(
        self,
        epoch: int,
        train_loss: float,
        train_acc: float,
        val_loss: Optional[float] = None,
        val_acc: Optional[float] = None,
        learning_rate: Optional[float] = None,
        reeig_eps_values: Optional[Dict[str, float]] = None,
        **kwargs,
    ):
        """Log metrics for a complete epoch
        
        Args:
            epoch: Current epoch number
            train_loss: Training loss
            train_acc: Training accuracy
            val_loss: Validation loss (optional)
            val_acc: Validation accuracy (optional)
            learning_rate: Current learning rate (optional)
            reeig_eps_values: Dictionary of learned epsilon values from ReEigLearnable layers (optional)
            **kwargs: Additional metrics to log
        """
        # Prepare training metrics
        train_metrics = {
            "train_loss": train_loss,
            "train_acc": train_acc,
            "learning_rate": learning_rate,
            "phase": "training",
            **kwargs,
        }
        
        # Add epsilon values if provided
        if reeig_eps_values:
            train_metrics["reeig_eps_values"] = reeig_eps_values
        
        # Log training metrics
        self.log_metrics(epoch, 0, train_metrics)

        # Log validation metrics if provided
        if val_loss is not None and val_acc is not None:
            self.log_metrics(
                epoch,
                0,
                {"val_loss": val_loss, "val_acc": val_acc, "phase": "validation"},
            )

        # Update current epoch
        self.update_status(
            {
                "current_epoch": epoch,
                "current_train_loss": train_loss,
                "current_train_acc": train_acc,
                "current_val_loss": val_loss,
                "current_val_acc": val_acc,
            }
        )

    def log_evaluation_metrics(
        self,
        test_loss: float,
        test_acc: float,
        classification_report: str,
        confusion_matrix: Optional[Any] = None,
        precision: Optional[float] = None,
        recall: Optional[float] = None,
        f1_score: Optional[float] = None,
        per_class_metrics: Optional[Dict[str, Any]] = None,
        input_cond_mean: Optional[float] = None,
        input_cond_std: Optional[float] = None,
        output_cond_mean: Optional[float] = None,
        output_cond_std: Optional[float] = None,
        **additional_metrics,
    ):
        """
        Log comprehensive evaluation metrics to a separate JSON file
        
        Args:
            test_loss: Test loss value
            test_acc: Test accuracy value
            classification_report: Classification report as string
            confusion_matrix: Confusion matrix (will be converted to list if numpy array)
            precision: Overall precision score
            recall: Overall recall score
            f1_score: Overall F1 score
            per_class_metrics: Dictionary with per-class F1, precision, recall
            input_cond_mean: Mean condition number at input covariance matrices
            input_cond_std: Std of condition number at input covariance matrices
            output_cond_mean: Mean condition number before LogEig layer
            output_cond_std: Std of condition number before LogEig layer
            **additional_metrics: Any additional metrics (ROC, AUC, energy, memory, etc.)
        """
        eval_file = self.log_dir / f"{self.experiment_name}_evaluation.json"
        
        # Prepare evaluation data
        eval_data = {
            "timestamp": time.time(),
            "test_loss": test_loss,
            "test_accuracy": test_acc,
            "precision": precision,
            "recall": recall,
            "f1_score": f1_score,
            "classification_report": classification_report,
        }
        
        # Add conditioning statistics
        if input_cond_mean is not None:
            eval_data["input_condition_mean"] = input_cond_mean
        if input_cond_std is not None:
            eval_data["input_condition_std"] = input_cond_std
        if output_cond_mean is not None:
            eval_data["output_condition_mean"] = output_cond_mean
        if output_cond_std is not None:
            eval_data["output_condition_std"] = output_cond_std
        
        # Convert confusion matrix to list if it's a numpy array
        if confusion_matrix is not None:
            if hasattr(confusion_matrix, "tolist"):
                eval_data["confusion_matrix"] = confusion_matrix.tolist()
            else:
                eval_data["confusion_matrix"] = confusion_matrix
        
        # Add per-class metrics
        if per_class_metrics is not None:
            eval_data["per_class_metrics"] = per_class_metrics
        
        # Add any additional metrics
        eval_data.update(additional_metrics)
        
        # Write to file
        with open(eval_file, "w") as f:
            json.dump(eval_data, f, indent=2)
        
        # Update status
        self.update_status({
            "test_loss": test_loss,
            "test_accuracy": test_acc,
            "evaluation_completed": True,
        })

    def finalize(self):
        """Mark experiment as completed and stop energy monitoring"""
        # Stop energy monitoring and get statistics
        energy_stats = self.stop_energy_monitoring()
        
        self.update_status({
            "status": "completed", 
            "end_time": time.time(),
            "energy_consumption": energy_stats,
        })