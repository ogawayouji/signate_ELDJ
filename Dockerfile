FROM python:3.8.2
USER root

RUN apt-get update \
  && apt-get upgrade -y \
  # imageのサイズを小さくするためにキャッシュ削除
  && apt-get clean \
  # && rm -rf /var/lib/apt/lists/* \
  && pip install --upgrade pip 
  # && apt-get install -y locale \
  # && locale-gen ja_JP.UTF-8
# RUN localedef -f UTF-8 -i ja_JP ja_JP.UTF-8

ENV LANG ja_JP.UTF-8
ENV LANGUAGE ja_JP:ja
ENV LC_ALL ja_JP.UTF-8
ENV TZ JST-9
# RUN localedef -f UTF-8 -i ja_JP ja_JP.utf8
ENV TERM xterm
# 作業ディレクトリ
WORKDIR /python_101/
RUN apt-get update
RUN apt -y upgrade 
RUN apt-get install -y vim less libmecab-dev
RUN pip install --upgrade setuptools

RUN apt-get install -y curl file git libmecab-dev sudo make mecab mecab-ipadic mecab-ipadic-utf8 swig xz-utils bash openssl wget
# RUN pip install mecab-python3
# RUN git clone https://github.com/taku910/mecab.git /mecab
# WORKDIR /mecab/mecab
# RUN git clone --depth 1 https://github.com/neologd/mecab-ipadic-neologd.git /mecab-ipadic-neologd && \
#     /mecab-ipadic-neologd/bin/install-mecab-ipadic-neologd -n -a -y -p /usr/local/lib/mecab/dic/mecab-ipadic-neologd
# RUN echo "dicdir = /usr/local/lib/mecab/dic/mecab-ipadic-neologd" > /usr/local/etc/mecabrc
# RUN rm -rf /mecab && \
#     rm -rf /mecab-ipadic-neologd
# WORKDIR /local
# RUN wget -O /local/CRF++-0.58.tar.gz 'https://drive.google.com/uc?export=download&id=0B4y35FiV1wh7QVR6VXJ5dWExSTQ' \
#     && cd /local/ \
#     && tar zxf CRF++-0.58.tar.gz \
#     && cd CRF++-0.58 \
#     && ./configure \
#     && make \
#     && make install

# Cabocha
# RUN cd /tmp \
#     && DOWNLOAD_URL="https://drive.google.com` curl -c cookies.txt \
#        'https://drive.google.com/uc?export=download&id=0B4y35FiV1wh7SDd1Q1dUQkZQaUU' \
#        | sed -r 's/"/\n/g' |grep id=0B4y35FiV1wh7SDd1Q1dUQkZQaUU |grep confirm |sed 's/&amp;/\&/g'`" \
#     && curl -L -b cookies.txt -o /tmp/cabocha-0.69.tar.bz2 "$DOWNLOAD_URL" \
#     && tar jxf cabocha-0.69.tar.bz2 \
#     && cd cabocha-0.69 \
#     && ./configure --with-mecab-config=`which mecab-config` --with-charset=utf8 \
#     && make \
#     && make install \
#     && cd python \
#     && python setup.py build \
#     && python setup.py install
# WORKDIR /local
# ENV CPPFLAGS -I/usr/local/include
# RUN curl -c  cabocha-0.69.tar.bz2 -s -L "https://drive.google.com/uc?export=download&id=0B4y35FiV1wh7SDd1Q1dUQkZQaUU" |grep confirm |  sed -e "s/^.*confirm=\(.*\)&amp;id=.*$/\1/" | xargs -I{} \
#     curl -b  cabocha-0.69.tar.bz2 -L -o cabocha-0.69.tar.bz2 "https://drive.google.com/uc?confirm={}&export=download&id=0B4y35FiV1wh7SDd1Q1dUQkZQaUU"
# RUN tar -jxf cabocha-0.69.tar.bz2
# WORKDIR cabocha-0.69
# RUN ./configure  --with-mecab-config=`which mecab-config` --with-charset=UTF8
# # --prefix=/usr/local
# RUN echo "/usr/local/lib" >> /etc/ld.so.conf.d/lib.conf
# RUN ldconfig
# RUN make
# RUN make install
# WORKDIR python
# RUN python setup.py build_ext
# RUN python setup.py install
# RUN ldconfig
# WORKDIR /local/cabocha-0.69
# RUN make
# RUN make install
# RUN echo "/etc/ld.so.conf.d/lib.conf" >> /usr/local/lib
# RUN ldconfig

# # LAPACK/BLAS (scikit-learnで必要)
# RUN cd /tmp \
#     && wget http://www.netlib.org/lapack/lapack-3.8.0.tar.gz \
#     && tar zxf lapack-3.8.0.tar.gz \
#     && cd lapack-3.8.0/ \
#     && cp make.inc.example make.inc \
#     && make blaslib \
#     && make lapacklib \
#     && cp librefblas.a /usr/lib/libblas.a \
#     && cp liblapack.a /usr/lib/liblapack.a \
#     && cd / \
#     && rm -rf /tmp/*

# RUN mkdir crf && curl -SL  https://drive.google.com/uc?export=download&id=0B4y35FiV1wh7QVR6VXJ5dWExSTQ \
#     | tar -zxvf crf/CRF++-0.58.tar.gz
# # -o 'crfpp.tar.gz'
# # RUN tar zxvf CRF++-0.58.tar.gz
# # 'crfpp.tar.gz'
# WORKDIR crf/CRF++-0.58
# RUN ./configure --enable-utf8-only && \
#     make && \
#     make install
# RUN apt-get install -y git bash curl openssl sudo make 

WORKDIR /python_101/
COPY requirements.txt ${PWD}
RUN pip install -r requirements.txt
WORKDIR /python_101/src

