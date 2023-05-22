import React from 'react'
import './style.css'

export default ({background, text, borderColor, size, solid=false}) => {
    
    return (
        !solid ? (
          <button style={{ background: background, borderColor: background }} className={`btn ${size}`}>
            {text}
          </button>
        ) : (
          <button style={{ background: background, borderColor: borderColor }} className={`btn ${size}`}>
            {text}
          </button>
        )
      )
}