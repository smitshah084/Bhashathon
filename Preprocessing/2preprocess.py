import os
import re
import logging
import argparse
import unicodedata
import hashlib
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
import random
from preprocess import MultilingualPreprocessor 
import time

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("preprocessing2.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def process_file(self, file_path, language, output_file):
    """Process a single file with all preprocessing steps."""
    logger.info(f"Processing {language} file: {file_path}")
    file_timing = {
        "html_removal": [],
        "whitespace_cleaning": [],
        "unicode_normalization": [],
        "script_normalization": [],
        "quality_filtering": [],
        "script_filtering": [],
        "total_processing": [],
        "file_sizes": {}
    }
    
    file_size = os.path.getsize(file_path)
    file_timing["file_sizes"][file_path] = file_size
    
    start_time = time.time()
    logger.info(f"File size: {file_size/1024/1024:.2f} MB")
    
    try:
        # Output file for processed text
        temp_file = output_file + ".temp"
        
        # Create an empty processed file
        with open(temp_file, 'w', encoding='utf-8') as out_f:
            pass
            
        # Process the file in chunks to save memory
        total_lines = 0
        processed_lines = 0
        skipped_lines = 0
        bytes_processed = 0
        
        chunk_size = 10000  # Process 10k lines at a time
        seen_hashes = set() if self.deduplicate else None
        
        # For progress and time estimation
        last_progress_time = time.time()
        progress_interval = 5  # seconds between progress updates
        
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as in_f:
            with open(temp_file, 'w', encoding='utf-8') as out_f:
                
                # Read and process file in chunks
                chunk = []
                for line in tqdm(in_f, desc=f"Processing {language}", unit="lines"):
                    total_lines += 1
                    bytes_processed += len(line.encode('utf-8'))
                    
                    # Add periodic progress logging with time estimation
                    current_time = time.time()
                    if current_time - last_progress_time > progress_interval:
                        elapsed = current_time - start_time
                        if bytes_processed > 0 and elapsed > 0:
                            bytes_per_sec = bytes_processed / elapsed
                            remaining_bytes = file_size - bytes_processed
                            eta_seconds = remaining_bytes / bytes_per_sec if bytes_per_sec > 0 else 0
                            
                            # Format ETA
                            if eta_seconds < 60:
                                eta_str = f"{eta_seconds:.1f} seconds"
                            elif eta_seconds < 3600:
                                eta_str = f"{eta_seconds/60:.1f} minutes"
                            else:
                                eta_str = f"{eta_seconds/3600:.1f} hours"
                                
                            logger.info(f"Progress: {bytes_processed/file_size*100:.1f}% - "
                                       f"Speed: {bytes_per_sec/1024:.1f} KB/sec - "
                                       f"ETA: {eta_str}")
                            
                        last_progress_time = current_time
                    
                    chunk.append(line)
                    if len(chunk) >= chunk_size:
                        processed, skipped, chunk_timing = self._process_chunk(chunk, out_f, language, seen_hashes)
                        processed_lines += processed
                        skipped_lines += skipped
                        
                        # Merge timing data
                        for key, values in chunk_timing.items():
                            if key != "file_sizes" and isinstance(values, list):
                                file_timing[key].extend(values)
                                
                        chunk = []
                
                # Process any remaining lines
                if chunk:
                    processed, skipped, chunk_timing = self._process_chunk(chunk, out_f, language, seen_hashes)
                    processed_lines += processed
                    skipped_lines += skipped
                    
                    # Merge timing data
                    for key, values in chunk_timing.items():
                        if key != "file_sizes" and isinstance(values, list):
                            file_timing[key].extend(values)
        
        total_time = time.time() - start_time
        processing_rate = file_size / total_time if total_time > 0 else 0
        
        logger.info(f"Completed processing {language}: {total_lines:,} lines read, "
                  f"{processed_lines:,} lines kept, {skipped_lines:,} lines skipped")
        logger.info(f"Processing rate: {processing_rate/1024:.2f} KB/sec")
        logger.info(f"Total processing time: {total_time:.2f} seconds")
        
        # Calculate and log step-wise timing
        step_times = {}
        for step, times in file_timing.items():
            if step != "file_sizes" and isinstance(times, list) and times:
                avg_time = sum(times) / len(times)
                total_step_time = sum(times)
                step_percent = (total_step_time / total_time) * 100 if total_time > 0 else 0
                step_times[step] = (avg_time, total_step_time, step_percent)
                
                logger.info(f"Step '{step}': {total_step_time:.2f} sec total, "
                          f"{avg_time*1000:.2f} ms avg, {step_percent:.1f}% of total time")
        
        # If we're sampling, perform the sampling now
        if self.sample_size and self.sample_size > 0:
            sampled_file = output_file
            self._sample_lines(temp_file, sampled_file, self.sample_size)
            os.remove(temp_file)
        else:
            # Just rename the temp file to the final output
            os.rename(temp_file, output_file)
        
        # Add processing rate to the file timing data
        file_timing["processing_rate"] = processing_rate
        file_timing["total_elapsed"] = total_time
        
        return True, file_timing
        
    except Exception as e:
        logger.error(f"Error processing {language} file: {str(e)}")
        return False, file_timing

def predict_processing_time(self, file_paths, base_rate=None):
    """
    Predict processing time for files based on their size and known processing rate.
    
    Args:
        file_paths: List of file paths to predict
        base_rate: Processing rate in bytes/second (if None, will use measured rate)
    
    Returns:
        Dictionary with estimated times for each file and total
    """
    if base_rate is None:
        # Calculate average processing rate from the timing stats
        if len(self.timing_stats["total_processing"]) > 0:
            total_bytes = sum(self.timing_stats["file_sizes"].values())
            total_time = sum(self.timing_stats["total_processing"])
            base_rate = total_bytes / total_time if total_time > 0 else 0
        else:
            # Default conservative estimate if no data available
            base_rate = 500 * 1024  # 500 KB/s
    
    predictions = {}
    total_size = 0
    total_time = 0
    
    for file_path in file_paths:
        if os.path.exists(file_path):
            file_size = os.path.getsize(file_path)
            total_size += file_size
            
            # Predict time for this file
            predicted_time = file_size / base_rate if base_rate > 0 else 0
            
            # Format time in appropriate units
            if predicted_time < 60:
                time_str = f"{predicted_time:.2f} seconds"
            elif predicted_time < 3600:
                time_str = f"{predicted_time/60:.2f} minutes"
            else:
                time_str = f"{predicted_time/3600:.2f} hours"
            
            predictions[file_path] = {
                "file_size": file_size,
                "estimated_time": predicted_time,
                "formatted_time": time_str
            }
            
            total_time += predicted_time
    
    # Format total time
    if total_time < 60:
        total_time_str = f"{total_time:.2f} seconds"
    elif total_time < 3600:
        total_time_str = f"{total_time/60:.2f} minutes"
    else:
        total_time_str = f"{total_time/3600:.2f} hours"
    
    # Add totals
    predictions["total"] = {
        "file_size": total_size,
        "estimated_time": total_time,
        "formatted_time": total_time_str,
        "processing_rate": f"{base_rate/1024:.2f} KB/sec"
    }
    
    return predictions

def generate_extended_timing_report(self):
    """Generate a comprehensive report with timing statistics and predictions for future runs."""
    # Create a timing report file
    report_path = os.path.join(self.output_dir, "timing_report.txt")
    
    # Calculate timing metrics
    total_bytes_processed = sum(self.timing_stats["file_sizes"].values())
    
    # Calculate averages for each processing step
    step_times = {}
    step_percentages = {}
    
    for step in ["html_removal", "whitespace_cleaning", "unicode_normalization", 
                "script_normalization", "quality_filtering", "script_filtering", 
                "total_processing"]:
        times = self.timing_stats[step]
        if times:
            avg_time = sum(times) / len(times)
            step_times[step] = avg_time
            
            # For percentage calculation (excluding total_processing)
            if step != "total_processing":
                step_percentages[step] = 0  # Will calculate after getting all steps
    
    # Calculate percentage of time spent in each step
    total_step_time = sum(step_times.values())
    if total_step_time > 0:
        for step in step_percentages:
            step_percentages[step] = (step_times[step] / total_step_time) * 100
    
    # Calculate processing rates
    processed_lines = len(self.timing_stats["total_processing"])
    bytes_per_second = 0
    lines_per_second = 0
    
    if processed_lines > 0:
        if sum(self.timing_stats["total_processing"]) > 0:
            bytes_per_second = total_bytes_processed / sum(self.timing_stats["total_processing"])
            lines_per_second = processed_lines / sum(self.timing_stats["total_processing"])
    
    # Generate predictions for various file sizes
    predictions = {}
    for size_mb in [1, 10, 100, 1000]:
        size_bytes = size_mb * 1024 * 1024
        if bytes_per_second > 0:
            est_time = size_bytes / bytes_per_second
            
            # Format time
            if est_time < 60:
                time_str = f"{est_time:.2f} seconds"
            elif est_time < 3600:
                time_str = f"{est_time/60:.2f} minutes"
            else:
                time_str = f"{est_time/3600:.2f} hours"
                
            predictions[size_mb] = time_str
    
    # Generate predictions for thread counts
    thread_predictions = {}
    if bytes_per_second > 0:
        base_size = 1 * 1024 * 1024 * 1024  # 1 GB
        for threads in [1, 2, 4, 8, 16, 32]:
            # Assume diminishing returns with more threads
            efficiency = min(1.0, 0.8 + (0.2 / threads))
            effective_rate = bytes_per_second * threads * efficiency
            est_time = base_size / effective_rate
            
            # Format time
            if est_time < 60:
                time_str = f"{est_time:.2f} seconds"
            elif est_time < 3600:
                time_str = f"{est_time/60:.2f} minutes"
            else:
                time_str = f"{est_time/3600:.2f} hours"
                
            thread_predictions[threads] = time_str
    
    # Write the report
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("============================================\n")
        f.write("   Multilingual Preprocessing Timing Report \n")
        f.write("============================================\n\n")
        
        f.write("PROCESSING SPEED METRICS\n")
        f.write("------------------------\n")
        f.write(f"Total bytes processed: {total_bytes_processed:,} bytes\n")
        f.write(f"Total lines processed: {processed_lines:,} lines\n")
        f.write(f"Processing rate: {bytes_per_second:.2f} bytes/sec ({bytes_per_second/1024:.2f} KB/sec)\n")
        f.write(f"Lines per second: {lines_per_second:.2f}\n")
        f.write(f"Average time per line: {(step_times.get('total_processing', 0)*1000):.2f} ms\n\n")
        
        f.write("PROCESSING STEP BREAKDOWN\n")
        f.write("------------------------\n")
        for step, percentage in sorted(step_percentages.items(), key=lambda x: x[1], reverse=True):
            f.write(f"{step:<20}: {percentage:6.2f}% ({step_times[step]*1000:.2f} ms avg)\n")
        f.write("\n")
        
        f.write("FILE DETAILS\n")
        f.write("------------\n")
        for file_path, file_size in self.timing_stats["file_sizes"].items():
            f.write(f"{os.path.basename(file_path):<30}: {file_size:,} bytes ({file_size/1024/1024:.2f} MB)\n")
        f.write("\n")
        
        f.write("ESTIMATED PROCESSING TIMES (SINGLE-THREADED)\n")
        f.write("-------------------------------------------\n")
        for size_mb, time_str in predictions.items():
            f.write(f"{size_mb:,} MB file: {time_str}\n")
        f.write("\n")
        
        f.write("ESTIMATED PROCESSING TIMES FOR 1 GB WITH DIFFERENT THREAD COUNTS\n")
        f.write("--------------------------------------------------------------\n")
        for threads, time_str in thread_predictions.items():
            f.write(f"{threads:2} threads: {time_str}\n")
        f.write("\n")
        
        f.write("OPTIMIZATION RECOMMENDATIONS\n")
        f.write("---------------------------\n")
        # Find the most time-consuming step
        if step_percentages:
            most_time_consuming = max(step_percentages.items(), key=lambda x: x[1])
            f.write(f"Most time-consuming step: '{most_time_consuming[0]}' ({most_time_consuming[1]:.2f}%)\n")
            
            # Give recommendations based on which step is slowest
            if most_time_consuming[0] == "unicode_normalization":
                f.write("Consider optimizing Unicode normalization by pre-caching common patterns\n")
            elif most_time_consuming[0] == "quality_filtering":
                f.write("Consider simplifying quality filtering criteria for speed improvement\n")
            elif most_time_consuming[0] == "script_filtering":
                f.write("Script filtering is expensive - consider pre-filtering files by language\n")
        
        f.write("\nRecommended thread count for your system: ")
        cpu_count = os.cpu_count() or 4
        f.write(f"{max(1, cpu_count - 1)} (based on available CPU cores)\n")
    
    logger.info(f"Extended timing report generated: {report_path}")
    
    # Log summary to console
    logger.info(f"Processing rate: {bytes_per_second/1024:.2f} KB/sec")
    logger.info(f"Average time per line: {(step_times.get('total_processing', 0)*1000):.2f} ms")
    
    # Return stats for further use
    return {
        "bytes_per_second": bytes_per_second,
        "lines_per_second": lines_per_second,
        "step_times": step_times,
        "step_percentages": step_percentages,
        "predictions": predictions,
        "thread_predictions": thread_predictions
    }

# Add this method to add real-time progress tracking during chunk processing
def _process_chunk_with_progress(self, chunk, out_file, language, seen_hashes=None):
    """Process a chunk of lines from the input file with detailed progress tracking."""
    processed_count = 0
    skipped_count = 0
    chunk_timing = {
        "html_removal": [],
        "whitespace_cleaning": [],
        "unicode_normalization": [],
        "script_normalization": [],
        "quality_filtering": [],
        "script_filtering": [],
        "total_processing": []
    }
    
    # Track bytes processed
    bytes_processed = 0
    
    # For very large chunks, show progress
    use_progress = len(chunk) > 1000
    iterator = tqdm(chunk, desc=f"Processing chunk", unit="lines") if use_progress else chunk
    
    chunk_start_time = time.time()
    
    for line in iterator:
        bytes_this_line = len(line.encode('utf-8'))
        bytes_processed += bytes_this_line
        
        start_time = time.time()
        processed_line, line_timing = self.preprocess_line(line, language)
        total_time = time.time() - start_time
        
        # Record timing for this line
        for key, value in line_timing.items():
            chunk_timing[key].append(value)
        chunk_timing["total_processing"].append(total_time)
        
        # Skip empty lines
        if not processed_line:
            skipped_count += 1
            continue
            
        # Optionally deduplicate using hashing
        if seen_hashes is not None:
            line_hash = hashlib.md5(processed_line.encode('utf-8')).hexdigest()
            if line_hash in seen_hashes:
                skipped_count += 1
                continue
            seen_hashes.add(line_hash)
        
        # Write the processed line
        out_file.write(processed_line + '\n')
        processed_count += 1
    
    # Calculate overall chunk statistics
    chunk_time = time.time() - chunk_start_time
    processing_rate = bytes_processed / chunk_time if chunk_time > 0 else 0
    
    if use_progress:
        logger.info(f"Chunk completed: {processed_count} kept, {skipped_count} skipped, "
                   f"rate: {processing_rate/1024:.2f} KB/sec")
    
    return processed_count, skipped_count, chunk_timing, processing_rate

# Add this to the main script to estimate time before starting full processing
def main():
    parser = argparse.ArgumentParser(description="Multilingual text preprocessing pipeline")
    parser.add_argument("--input", nargs='+', help="Input text files to process")
    parser.add_argument("--output-dir", default="preprocessed", help="Output directory for processed files")
    parser.add_argument("--threads", type=int, default=4, help="Number of threads to use")
    parser.add_argument("--sample-size", type=int, default=None, help="Sample N lines from each processed file")
    parser.add_argument("--estimate", action="store_true", help="Estimate processing time without full processing")
    parser.add_argument("--no-deduplicate", action="store_true", help="Skip deduplication")
    parser.add_argument("--no-unicode-clean", action="store_true", help="Skip unicode normalization")
    parser.add_argument("--no-quality-filter", action="store_true", help="Skip quality filtering")
    parser.add_argument("--no-script-filter", action="store_true", help="Skip script filtering")
    
    args = parser.parse_args()
    
    if not args.input:
        logger.error("No input files specified. Use --input to specify files.")
        return
    
    # Configure the preprocessor
    preprocessor = MultilingualPreprocessor(
        input_files=args.input,
        output_dir=args.output_dir,
        num_threads=args.threads,
        sample_size=args.sample_size,
        deduplicate=not args.no_deduplicate,
        clean_unicode=not args.no_unicode_clean,
        remove_low_quality=not args.no_quality_filter,
        script_filtering=not args.no_script_filter
    )
    
    # Check if files exist
    valid_files = [f for f in args.input if os.path.exists(f)]
    if not valid_files:
        logger.error("None of the specified input files exist.")
        return
    
    # If estimate mode, run quick estimation and exit
    if args.estimate:
        logger.info("Running time estimation...")
        
        # Estimate based on a sample
        sample_size = 1000  # lines to sample for estimation
        
        # Calculate file sizes
        total_size = 0
        file_info = {}
        
        for file_path in valid_files:
            size = os.path.getsize(file_path)
            file_info[file_path] = {"size": size, "size_mb": size/1024/1024}
            total_size += size
            
            logger.info(f"File: {os.path.basename(file_path)}, Size: {size/1024/1024:.2f} MB")
        
        logger.info(f"Total size: {total_size/1024/1024:.2f} MB")
        
        # Process samples from each file to get average processing rate
        sampled_lines = []
        for file_path in valid_files:
            try:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    # Count total lines in file (up to a reasonable limit)
                    line_count = 0
                    for _ in f:
                        line_count += 1
                        if line_count >= 100000:  # Stop counting at 100k lines
                            break
                
                # Get actual line count if didn't break early
                if line_count < 100000:
                    file_info[file_path]["line_count"] = line_count
                else:
                    # Estimate based on first 100k lines
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                        first_100k = [next(f) for _ in range(100000)]
                    avg_line_size = sum(len(line.encode('utf-8')) for line in first_100k) / 100000
                    est_line_count = int(file_info[file_path]["size"] / avg_line_size)
                    file_info[file_path]["line_count"] = est_line_count
                    file_info[file_path]["line_count_estimated"] = True
                
                # Sample random lines
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    lines = f.readlines()
                    
                    # If file is small enough, use all lines, otherwise sample
                    if len(lines) <= sample_size:
                        sampled_lines.extend(lines)
                    else:
                        base_name = os.path.basename(file_path)
                        language = preprocessor.file_to_lang.get(base_name, Path(file_path).stem)
                        # Weight the sample toward the beginning since it's often cleaner
                        start_lines = lines[:min(1000, len(lines)//10)]
                        remaining_lines = lines[min(1000, len(lines)//10):]
                        
                        # Take some from start and some random from rest
                        start_sample = min(sample_size // 5, len(start_lines))
                        random_sample = min(sample_size - start_sample, len(remaining_lines))
                        
                        sampled_lines.extend(start_lines[:start_sample])
                        sampled_lines.extend(random.sample(remaining_lines, random_sample))
                        
                        logger.info(f"Sampled {start_sample + random_sample} lines from {language} file")
                
            except Exception as e:
                logger.error(f"Error sampling {file_path}: {str(e)}")
        
        # Process the sampled lines to estimate rate
        logger.info(f"Processing {len(sampled_lines)} sampled lines to estimate rate...")
        
        # Time the processing
        total_sample_bytes = sum(len(line.encode('utf-8')) for line in sampled_lines)
        start_time = time.time()
        
        # Process each line
        processed_lines = 0
        for line in tqdm(sampled_lines, desc="Processing samples"):
            base_name = os.path.basename(valid_files[0])  # Use first file's language as fallback
            language = preprocessor.file_to_lang.get(base_name, "Unknown")
            processed_line = preprocessor.preprocess_line(line, language)
            if processed_line:
                processed_lines += 1
        
        sample_time = time.time() - start_time
        
        # Calculate processing rate
        if sample_time > 0:
            bytes_per_second = total_sample_bytes / sample_time
            
            logger.info(f"Sample processing rate: {bytes_per_second/1024:.2f} KB/sec")
            logger.info(f"Processed {processed_lines} of {len(sampled_lines)} sampled lines")
            logger.info(f"Line retention rate: {processed_lines / len(sampled_lines) * 100:.1f}%")
            
            # Estimate times for each file
            logger.info("\nEstimated processing times:")
            total_est_time = 0
            
            for file_path, info in file_info.items():
                est_time = info["size"] / bytes_per_second
                total_est_time += est_time
                
                # Format time
                if est_time < 60:
                    time_str = f"{est_time:.2f} seconds"
                elif est_time < 3600:
                    time_str = f"{est_time/60:.2f} minutes"
                else:
                    time_str = f"{est_time/3600:.2f} hours"
                
                logger.info(f"{os.path.basename(file_path)}: {time_str} " +
                           f"({info['size_mb']:.2f} MB, ~{info['line_count']:,} lines)")
            
            # Format total time
            if total_est_time < 60:
                time_str = f"{total_est_time:.2f} seconds"
            elif total_est_time < 3600:
                time_str = f"{total_est_time/60:.2f} minutes"
            else:
                time_str = f"{total_est_time/3600:.2f} hours"
                
            logger.info(f"\nTotal estimated time: {time_str}")
            
            # Estimate with different thread counts
            logger.info("\nEstimated times with different thread counts:")
            for threads in [1, 2, 4, 8, 16]:
                # Assume diminishing returns with more threads
                efficiency = 1 if threads == 1 else 0.8
                thread_time = total_est_time / (threads * efficiency)
                
                # Format time
                if thread_time < 60:
                    time_str = f"{thread_time:.2f} seconds"
                elif thread_time < 3600:
                    time_str = f"{thread_time/60:.2f} minutes"
                else:
                    time_str = f"{thread_time/3600:.2f} hours"
                
                logger.info(f"{threads} threads: {time_str}")
        
        else:
            logger.error("Sample processing took no time. Unable to estimate.")
        
        return
    
    # Regular processing
    logger.info(f"Starting preprocessing with {args.threads} threads...")
    processed_files = preprocessor.process_all_files()
    
    # Generate timing report
    preprocessor.generate_extended_timing_report()
    
    logger.info(f"Preprocessing complete. Files: {processed_files}")

if __name__ == "__main__":
    main()
