#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
VideoCutter 테스트 및 사용 예시
"""

import os
from video_cutter import VideoCutter

def test_basic_functionality():
    """기본 기능 테스트"""
    print("=== 기본 기능 테스트 ===")
    
    # VideoCutter 인스턴스 생성
    cutter = VideoCutter(output_dir="test_output")
    
    # 비디오 정보 가져오기 (테스트용)
    test_video = "test_video.mp4"  # 실제 비디오 파일 경로로 변경하세요
    
    if os.path.exists(test_video):
        try:
            info = cutter.get_video_info(test_video)
            print(f"비디오 정보: {info}")
        except Exception as e:
            print(f"비디오 정보 가져오기 실패: {e}")
    
    print("✅ 기본 기능 테스트 완료\n")

def test_single_segment():
    """단일 구간 자르기 테스트"""
    print("=== 단일 구간 자르기 테스트 ===")
    
    cutter = VideoCutter(output_dir="test_output")
    test_video = "test_video.mp4"  # 실제 비디오 파일 경로로 변경하세요
    
    if os.path.exists(test_video):
        try:
            # 10초부터 25초까지 구간 자르기
            output_path = cutter.cut_segment(
                input_path=test_video,
                start_time=10.0,
                end_time=25.0,
                output_path="test_output/single_cut.mp4",
                codec="copy",  # 빠른 복사
                quality="high"
            )
            print(f"✅ 단일 구간 자르기 성공: {output_path}")
        except Exception as e:
            print(f"❌ 단일 구간 자르기 실패: {e}")
    else:
        print("⚠️ 테스트 비디오 파일이 없습니다. 경로를 확인하세요.")
    
    print()

def test_multiple_segments():
    """여러 구간 자르기 테스트"""
    print("=== 여러 구간 자르기 테스트 ===")
    
    cutter = VideoCutter(output_dir="test_output")
    test_video = "test_video.mp4"  # 실제 비디오 파일 경로로 변경하세요
    
    if os.path.exists(test_video):
        try:
            # 여러 구간 정의
            segments = [
                (10.0, 25.0),   # 10초-25초
                (45.0, 60.0),   # 45초-60초
                (90.0, 105.0),  # 90초-105초
            ]
            
            output_paths = cutter.cut_multiple_segments(
                input_path=test_video,
                segments=segments,
                output_dir="test_output/multiple_cuts",
                codec="copy",
                quality="high"
            )
            
            print(f"✅ 여러 구간 자르기 성공: {len(output_paths)}개 파일 생성")
            for path in output_paths:
                print(f"  - {path}")
                
        except Exception as e:
            print(f"❌ 여러 구간 자르기 실패: {e}")
    else:
        print("⚠️ 테스트 비디오 파일이 없습니다. 경로를 확인하세요.")
    
    print()

def test_from_spans():
    """Span 데이터에서 구간 자르기 테스트"""
    print("=== Span 데이터에서 구간 자르기 테스트 ===")
    
    cutter = VideoCutter(output_dir="test_output")
    test_video = "test_video.mp4"  # 실제 비디오 파일 경로로 변경하세요
    
    if os.path.exists(test_video):
        try:
            # 예시 span 데이터 (실제로는 VTL 결과에서 가져옴)
            spans = [
                {"t0": 10.5, "t1": 25.3, "conf": 0.95, "tag": "person_walking"},
                {"t0": 45.2, "t1": 60.1, "conf": 0.87, "tag": "person_sitting"},
                {"t0": 90.0, "t1": 105.5, "conf": 0.92, "tag": "person_eating"},
            ]
            
            output_paths = cutter.cut_from_spans(
                input_path=test_video,
                spans=spans,
                output_dir="test_output/span_cuts",
                codec="copy",
                quality="high"
            )
            
            print(f"✅ Span에서 구간 자르기 성공: {len(output_paths)}개 파일 생성")
            for i, path in enumerate(output_paths):
                span = spans[i]
                print(f"  - {span['tag']}: {span['t0']:.1f}s-{span['t1']:.1f}s → {path}")
                
        except Exception as e:
            print(f"❌ Span에서 구간 자르기 실패: {e}")
    else:
        print("⚠️ 테스트 비디오 파일이 없습니다. 경로를 확인하세요.")
    
    print()

def test_integration_with_vtl():
    """VTL 시스템과의 통합 예시"""
    print("=== VTL 시스템과의 통합 예시 ===")
    
    # VTL 실행 결과에서 얻은 spans 예시
    vtl_result_spans = [
        {"t0": 15.2, "t1": 28.7, "conf": 0.94, "source": "probe_result"},
        {"t0": 52.1, "t1": 67.3, "conf": 0.89, "source": "sequence_result"},
        {"t0": 88.5, "t1": 102.1, "conf": 0.91, "source": "join_result"},
    ]
    
    print("VTL 결과 spans:")
    for i, span in enumerate(vtl_result_spans):
        print(f"  {i+1}. {span['source']}: {span['t0']:.1f}s-{span['t1']:.1f}s (conf: {span['conf']:.2f})")
    
    # VideoCutter로 구간 자르기
    cutter = VideoCutter(output_dir="vtl_output")
    
    try:
        # 실제 비디오 파일이 있다면 실행
        test_video = "test_video.mp4"
        if os.path.exists(test_video):
            output_paths = cutter.cut_from_spans(
                input_path=test_video,
                spans=vtl_result_spans,
                output_dir="vtl_output/segments",
                codec="copy",
                quality="high"
            )
            
            print(f"\n✅ VTL 결과 구간 자르기 성공: {len(output_paths)}개 파일 생성")
            for i, path in enumerate(output_paths):
                span = vtl_result_spans[i]
                print(f"  - {span['source']}: {path}")
        else:
            print("\n⚠️ 테스트 비디오 파일이 없어서 실제 자르기는 건너뜁니다.")
            
    except Exception as e:
        print(f"\n❌ VTL 통합 테스트 실패: {e}")
    
    print()

def main():
    """메인 테스트 함수"""
    print("🎬 VideoCutter 테스트 시작\n")
    
    try:
        test_basic_functionality()
        test_single_segment()
        test_multiple_segments()
        test_from_spans()
        test_integration_with_vtl()
        
        print("🎉 모든 테스트 완료!")
        
    except Exception as e:
        print(f"❌ 테스트 중 오류 발생: {e}")

if __name__ == "__main__":
    main()
