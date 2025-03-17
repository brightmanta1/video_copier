import { Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Observable } from 'rxjs';

@Injectable({
  providedIn: 'root'
})
export class VideoService {
  private apiUrl = 'http://localhost:5000/api';

  constructor(private http: HttpClient) { }

  /**
   * Отправляет видео на анализ
   */
  analyzeVideo(videoFile: File): Observable<any> {
    const formData = new FormData();
    formData.append('video', videoFile);
    return this.http.post(`${this.apiUrl}/analyze`, formData);
  }

  /**
   * Применяет структуру редактирования к новому видео
   */
  applyStructure(videoFile: File, structure: any): Observable<any> {
    const formData = new FormData();
    formData.append('video', videoFile);
    formData.append('structure', JSON.stringify(structure));
    return this.http.post(`${this.apiUrl}/apply-structure`, formData);
  }

  /**
   * Проверка соединения с сервером
   */
  getHealthStatus(): Observable<any> {
    return this.http.get(`${this.apiUrl}/health`);
  }
  
  /**
   * Получает полный URL для доступа к результирующему видео
   */
  getFullUrl(relativePath: string): string {
    // Убираем начальный слеш, если он есть
    const path = relativePath.startsWith('/') ? relativePath.substring(1) : relativePath;
    return `${this.apiUrl}/${path}`;
  }
} 