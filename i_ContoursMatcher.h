#ifndef I_CONTOURSMATCHER_H
#define I_CONTOURSMATCHER_H


template<typename VT, typename MT>
class i_ContoursMatcher
{
public:
    explicit i_ContoursMatcher();

    virtual void setContours(VT contours)=0;
    virtual float estimateDiff(MT image)=0;
};

#endif // I_CONTOURSMATCHER_H
