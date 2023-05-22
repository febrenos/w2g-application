import React from "react";
import Planet from '../../img/g1.png'
import * as Styled from './style.jsx';


export default function({width=400}){
  return(
    <Styled.Content width={width} height={width}>
        <Styled.Planet src={Planet} width={width} />
    </Styled.Content>
  )
}