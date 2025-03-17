import { Component, OnInit } from '@angular/core';
import { FormBuilder, FormGroup } from '@angular/forms';
import { MatSnackBar } from '@angular/material/snack-bar';
import { VideoService } from '../../services/video.service';

@Component({
  selector: 'app-video-upload',
  templateUrl: './video-upload.component.html',
  styleUrls: ['./video-upload.component.scss']
})
export class VideoUploadComponent implements OnInit {
  uploadForm: FormGroup;
  sourceVideo: File | null = null;
  sourceVideoUrl: string | null = null;
  targetVideo: File | null = null;
  targetVideoUrl: string | null = null;
  
  isLoading = false;
  videoAnalysis: any = null;
  resultVideoUrl: string | null = null;
  
  constructor(
    private fb: FormBuilder,
    private videoService: VideoService,
    private snackBar: MatSnackBar
  ) {
    this.uploadForm = this.fb.group({
      sourceVideo: [null],
      targetVideo: [null]
    });
  }

  ngOnInit(): void {
    // Проверяем подключение к серверу
    this.checkServerConnection();
  }
  
  checkServerConnection(): void {
    this.videoService.getHealthStatus().subscribe(
      response => {
        console.log('Server connection established');
      },
      error => {
        this.snackBar.open('Не удалось подключиться к серверу. Проверьте запущен ли сервер.', 'Закрыть', {
          duration: 5000
        });
      }
    );
  }

  onSourceVideoSelected(event: any): void {
    if (event.target.files && event.target.files.length) {
      this.sourceVideo = event.target.files[0];
      this.uploadForm.patchValue({ sourceVideo: this.sourceVideo });
      
      // Создаем URL для предпросмотра
      if (this.sourceVideoUrl) {
        URL.revokeObjectURL(this.sourceVideoUrl);
      }
      this.sourceVideoUrl = URL.createObjectURL(this.sourceVideo);
    }
  }
  
  onTargetVideoSelected(event: any): void {
    if (event.target.files && event.target.files.length) {
      this.targetVideo = event.target.files[0];
      this.uploadForm.patchValue({ targetVideo: this.targetVideo });
      
      // Создаем URL для предпросмотра
      if (this.targetVideoUrl) {
        URL.revokeObjectURL(this.targetVideoUrl);
      }
      this.targetVideoUrl = URL.createObjectURL(this.targetVideo);
    }
  }

  analyzeSourceVideo(): void {
    if (!this.sourceVideo) {
      this.snackBar.open('Сначала выберите исходное видео', 'Закрыть', {
        duration: 3000
      });
      return;
    }

    this.isLoading = true;
    this.videoService.analyzeVideo(this.sourceVideo).subscribe(
      data => {
        this.videoAnalysis = data;
        this.isLoading = false;
        this.snackBar.open('Анализ видео успешно завершен', 'Закрыть', {
          duration: 3000
        });
      },
      error => {
        this.isLoading = false;
        this.snackBar.open('Ошибка при анализе видео', 'Закрыть', {
          duration: 3000
        });
        console.error('Error analyzing video:', error);
      }
    );
  }

  applyToTargetVideo(): void {
    if (!this.targetVideo) {
      this.snackBar.open('Сначала выберите целевое видео', 'Закрыть', {
        duration: 3000
      });
      return;
    }

    if (!this.videoAnalysis) {
      this.snackBar.open('Сначала выполните анализ исходного видео', 'Закрыть', {
        duration: 3000
      });
      return;
    }

    this.isLoading = true;
    this.videoService.applyStructure(this.targetVideo, this.videoAnalysis).subscribe(
      data => {
        this.isLoading = false;
        if (data && data.result_url) {
          this.resultVideoUrl = this.videoService.getFullUrl(data.result_url);
          this.snackBar.open('Структура успешно применена', 'Закрыть', {
            duration: 3000
          });
        }
      },
      error => {
        this.isLoading = false;
        this.snackBar.open('Ошибка при применении структуры', 'Закрыть', {
          duration: 3000
        });
        console.error('Error applying structure:', error);
      }
    );
  }

  resetAll(): void {
    // Сбрасываем все состояния
    this.uploadForm.reset();
    this.sourceVideo = null;
    this.targetVideo = null;
    
    if (this.sourceVideoUrl) {
      URL.revokeObjectURL(this.sourceVideoUrl);
      this.sourceVideoUrl = null;
    }
    
    if (this.targetVideoUrl) {
      URL.revokeObjectURL(this.targetVideoUrl);
      this.targetVideoUrl = null;
    }
    
    this.videoAnalysis = null;
    this.resultVideoUrl = null;
  }
} 