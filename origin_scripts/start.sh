#!/bin/bash

echo "��ȡ��ɫ���ơ�ԭʼͼ���CC_xml��"
python scripts/pull_ourdata.py
REM ��ͣ3���Կ������
sleep 3
echo "��ȡ���ݣ�ת��Ϊcolmap��ʽ"
python scripts/convey2ourdata.py
REM ��ͣ3���Կ������
sleep 3
echo "���������Ϣ�У��˹��̺�ʱ�ϳ�"
python scripts/c2d_new.py
echo "��ɣ����ֶ�������Ϊ���ݼ��Ľ�����������xx/sparse/0�У�������Ϊ��points3D.ply"