@echo off

echo "��ȡ��ɫ���ơ�ԭʼͼ���CC_xml��"
python scripts/pull_ourdata.py

echo "��ȡ���ݣ�ת��Ϊcolmap��ʽ"
@REM REM ��ͣ3���Կ������
timeout /t 3 /nobreak
python scripts/convey2ourdata.py

echo "���������Ϣ�У��˹��̺�ʱ�ϳ�"
@REM REM ��ͣ3���Կ������
timeout /t 3 /nobreak

python scripts/c2d_new.py
echo "��ɣ����ֶ�������Ϊ���ݼ��Ľ�����������xx/sparse/0�У�������Ϊ��points3D.ply"